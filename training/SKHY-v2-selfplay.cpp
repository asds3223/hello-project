#include <algorithm>
#include <array>
#include <atomic>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <mutex>
#include <random>
#include <string>
#include <thread>
#include <vector>
#include "core/position.hpp"
#include "rules/renju.hpp"
#include "search/search.hpp"

using namespace nnue1skhy;

#pragma pack(push,1)
struct RawSample {
    uint64_t black[4];
    uint64_t white[4];
    int16_t searchScore;
    int8_t result;
    uint8_t side;
    uint16_t ply;
    uint16_t bestSq;
};
struct AuxSample {
    uint16_t finalPly;
    uint8_t startPly;
    uint8_t flags;
};
struct TeacherSampleV2 {
    uint16_t topSq[4];
    int16_t topScore[4];
    uint8_t topCount;
    uint8_t depth;
    uint16_t flags;       // bit0 uncertainty, bit1 short2, bit2 short6
    uint32_t nodes;
    int16_t margin;
    int16_t short2;
    int16_t short6;
    uint16_t uncertainty;
};
#pragma pack(pop)
static_assert(sizeof(RawSample)==72, "training sample format changed");
static_assert(sizeof(AuxSample)==4, "training aux format changed");
static_assert(sizeof(TeacherSampleV2)==32, "teacher v2 sidecar format changed");

struct PendingSample {
    RawSample s{};
    TeacherSampleV2 t{};
    Color stm=Color::Black;
    uint8_t flags=0;
};

static constexpr int kMateThreshold = SCORE_WIN - 512;
static constexpr int kStoredScoreLimit = 30000;

static int16_t encode_teacher_score(int score) {
    if(score >= kMateThreshold) return static_cast<int16_t>(std::clamp(score, kMateThreshold, kStoredScoreLimit));
    if(score <= -kMateThreshold) return static_cast<int16_t>(std::clamp(score, -kStoredScoreLimit, -kMateThreshold));
    return static_cast<int16_t>(std::clamp(score, -kStoredScoreLimit, kStoredScoreLimit));
}

static bool game_won(const Position& p) {
    if(p.last_move()==NO_SQ) return false;
    return p.last_color()==Color::Black
        ? RenjuRules::has_exact_five(p,p.last_move(),Color::Black)
        : RenjuRules::has_five(p,p.last_move(),Color::White);
}

static std::vector<Move> legal(Position& p) {
    auto m=p.generate_candidates(2); std::vector<Move> out; out.reserve(m.size());
    for(auto x:m){
        if(p.side_to_move()==Color::Black){
            auto l=RenjuRules::classify_move(p,x,Color::Black);
            if(l==MoveLegality::Legal || l==MoveLegality::Win) out.push_back(x);
        } else out.push_back(x);
    }
    return out;
}

static int static_move_score(const Position& p, Move m) {
    const Color us=p.side_to_move();
    const Color them=us==Color::Black?Color::White:Color::Black;
    const int x=m.x(), y=m.y();
    const int center=14-(std::abs(x-7)+std::abs(y-7));
    return p.neighbor_count2(m.sq)*20
         + p.line_neighbor_count4(m.sq,us)*13
         + p.line_neighbor_count4(m.sq,them)*11
         + center;
}

static void sort_static(const Position& p, std::vector<Move>& moves) {
    std::stable_sort(moves.begin(),moves.end(),[&](Move a,Move b){
        const int sa=static_move_score(p,a), sb=static_move_score(p,b);
        if(sa!=sb) return sa>sb;
        return a.sq<b.sq;
    });
}

static int probe_move(Searcher& searcher, Position& p, Move m, int probeMs, int probeDepth) {
    if(!p.make_move(m)) return -SCORE_INF;
    int v;
    if(game_won(p)) v=SCORE_WIN-1;
    else {
        SearchLimits lim; lim.maxDepth=std::max(1,probeDepth); lim.moveTimeMs=std::max(1,probeMs);
        auto r=searcher.search(p,lim);
        v=-r.score;
    }
    p.undo_move();
    return v;
}

struct Choice {
    Move played{};
    Move teacherBest{};
    int teacherScore=0;
    int teacherDepth=0;
    uint64_t teacherNodes=0;
    std::vector<Move> pv;
    std::array<Move,4> topMoves{};
    std::array<int,4> topScores{};
    int topCount=0;
    bool randomized=false;
};

static Choice searched_choice(Searcher& searcher, Position& p, std::mt19937& rng,
                              int depth, int movetime,
                              double topkProb, int topk, int margin,
                              int probeCandidates, int probeMs, int probeDepth,
                              int topkMinPly, int topkMaxPly,
                              int recordTopK) {
    SearchLimits lim; lim.maxDepth=depth; lim.moveTimeMs=movetime;
    auto main=searcher.search(p,lim);
    Choice out; out.played=main.bestMove; out.teacherBest=main.bestMove; out.teacherScore=main.score;
    out.teacherDepth=main.depth; out.teacherNodes=main.nodes; out.pv=main.pv;
    if(!out.played.valid()) return out;

    const bool randomEligible=topkProb>0.0 && topk>1 && p.ply()>=topkMinPly && p.ply()<=topkMaxPly;
    const bool wantsRandom=randomEligible && std::generate_canonical<double,32>(rng)<topkProb;
    const int needTop=std::clamp(std::max(recordTopK,wantsRandom?topk:1),1,4);
    if(needTop<=1){ out.topMoves[0]=main.bestMove; out.topScores[0]=main.score; out.topCount=1; return out; }

    auto lm=legal(p);
    if(lm.empty()) return out;
    sort_static(p,lm);
    std::vector<Move> candidates; candidates.reserve(static_cast<size_t>(std::max(2,probeCandidates)));
    candidates.push_back(main.bestMove);
    for(Move m:lm){
        if(m.sq==main.bestMove.sq) continue;
        candidates.push_back(m);
        if(static_cast<int>(candidates.size())>=std::max(2,probeCandidates)) break;
    }

    struct P { Move m; int s; };
    std::vector<P> scored; scored.reserve(candidates.size());
    for(Move m:candidates) scored.push_back({m,m.sq==main.bestMove.sq?main.score:probe_move(searcher,p,m,probeMs,probeDepth)});
    std::stable_sort(scored.begin(),scored.end(),[](const P&a,const P&b){if(a.s!=b.s)return a.s>b.s;return a.m.sq<b.m.sq;});
    out.topCount=std::min<int>(needTop,scored.size());
    for(int i=0;i<out.topCount;++i){out.topMoves[static_cast<size_t>(i)]=scored[static_cast<size_t>(i)].m;out.topScores[static_cast<size_t>(i)]=scored[static_cast<size_t>(i)].s;}

    if(!wantsRandom) return out;
    const int randomCap=std::min<int>(topk,scored.size());
    if(randomCap<2) return out;
    const int bestProbe=scored.front().s;
    std::vector<P> near;
    for(int i=0;i<randomCap;++i) if(scored[static_cast<size_t>(i)].s>=bestProbe-margin) near.push_back(scored[static_cast<size_t>(i)]);
    if(near.size()<2) return out;
    std::vector<double> weights; weights.reserve(near.size());
    const double temp=std::max(64.0,static_cast<double>(margin)/2.0);
    for(const auto& q:near) weights.push_back(std::exp(static_cast<double>(q.s-bestProbe)/temp));
    std::discrete_distribution<size_t> pick(weights.begin(),weights.end());
    Move chosen=near[pick(rng)].m;
    if(chosen.sq!=main.bestMove.sq){ out.played=chosen; out.randomized=true; }
    return out;
}

static bool pv_future_probe(Searcher& searcher, Position& p, const std::vector<Move>& pv,
                            int prefixPlies, int probeMs, int probeDepth, int& scoreOut) {
    if(prefixPlies<=0 || static_cast<int>(pv.size())<prefixPlies) return false;
    int made=0;
    for(int i=0;i<prefixPlies;++i){
        Move m=pv[static_cast<size_t>(i)];
        if(!m.valid() || !p.make_move(m)) break;
        ++made;
        if(game_won(p)) break;
    }
    const bool ok=(made==prefixPlies && !game_won(p));
    if(ok){
        SearchLimits lim; lim.maxDepth=std::max(1,probeDepth); lim.moveTimeMs=std::max(1,probeMs);
        auto r=searcher.search(p,lim);
        scoreOut=(prefixPlies&1)?-r.score:r.score;
    }
    while(made-->0) p.undo_move();
    return ok;
}

static bool build_midgame_seed(Position& p, Searcher& searcher, std::mt19937& rng,
                               int targetPly, int seedMs, int seedDepth, int staticTopK) {
    for(int attempt=0;attempt<8;++attempt){
        p.clear(); bool failed=false;
        for(int t=0;t<targetPly;++t){
            auto lm=legal(p); if(lm.empty()){failed=true;break;}
            sort_static(p,lm);
            Move chosen;
            if(t<2 || seedMs<=0){
                const size_t cap=std::min<size_t>(lm.size(),static_cast<size_t>(std::max(1,staticTopK)));
                chosen=lm[std::uniform_int_distribution<size_t>(0,cap-1)(rng)];
            } else {
                SearchLimits lim; lim.maxDepth=std::max(1,seedDepth); lim.moveTimeMs=std::max(1,seedMs);
                auto r=searcher.search(p,lim);
                chosen=r.bestMove.valid()?r.bestMove:lm.front();
                if(lm.size()>1 && std::uniform_int_distribution<int>(0,99)(rng)<35){
                    const size_t cap=std::min<size_t>(lm.size(),static_cast<size_t>(std::max(2,staticTopK)));
                    chosen=lm[std::uniform_int_distribution<size_t>(0,cap-1)(rng)];
                }
            }
            if(!chosen.valid() || !p.make_move(chosen)){failed=true;break;}
            if(game_won(p)){failed=true;break;}
        }
        if(!failed && p.ply()==targetPly) return true;
    }
    p.clear(); return false;
}

int main(int argc,char**argv){
    int games=1, depth=5, movetime=100, workers=0, hashMb=32, searchThreads=1;
    int openingRandomPlies=4, recordStride=1, minRecordPly=0, maxRecordPly=225;
    int topk=3, topkMargin=900, topkProbeCandidates=6, topkProbeMs=8, topkProbeDepth=5, topkMinPly=4, topkMaxPly=40;
    double topkRandomProb=0.18;
    double midgameStartProb=0.18;
    int midgameStartMinPly=12, midgameStartMaxPly=26, midgameSeedMs=6, midgameSeedDepth=4, midgameSeedStaticTopK=5;
    int policyRecordTopK=4;
    double uncertaintyProbeProb=0.20, shortTargetProbeProb=0.15;
    int uncertaintyProbeMs=6, uncertaintyProbeDepth=4, shortProbeMs=4, shortProbeDepth=3;
    uint32_t seed=0x1A5B7C9Du;
    std::string outPath="data/selfplay_00000.bin", auxPath, teacherPath;
    std::string nnuePath;
    bool scalarNnue=false;
    for(int i=1;i<argc;i++){
        std::string a=argv[i];
        if(a=="--games"&&i+1<argc) games=std::max(1,std::stoi(argv[++i]));
        else if(a=="--depth"&&i+1<argc) depth=std::max(1,std::stoi(argv[++i]));
        else if(a=="--movetime"&&i+1<argc) movetime=std::max(1,std::stoi(argv[++i]));
        else if(a=="--workers"&&i+1<argc) workers=std::max(0,std::stoi(argv[++i]));
        else if(a=="--hash-mb"&&i+1<argc) hashMb=std::max(1,std::stoi(argv[++i]));
        else if(a=="--search-threads"&&i+1<argc) searchThreads=std::max(1,std::stoi(argv[++i]));
        else if(a=="--opening-random-plies"&&i+1<argc) openingRandomPlies=std::clamp(std::stoi(argv[++i]),0,12);
        else if(a=="--record-stride"&&i+1<argc) recordStride=std::max(1,std::stoi(argv[++i]));
        else if(a=="--min-record-ply"&&i+1<argc) minRecordPly=std::clamp(std::stoi(argv[++i]),0,225);
        else if(a=="--max-record-ply"&&i+1<argc) maxRecordPly=std::clamp(std::stoi(argv[++i]),0,225);
        else if(a=="--topk-random-prob"&&i+1<argc) topkRandomProb=std::clamp(std::stod(argv[++i]),0.0,1.0);
        else if(a=="--topk"&&i+1<argc) topk=std::clamp(std::stoi(argv[++i]),1,12);
        else if(a=="--topk-margin"&&i+1<argc) topkMargin=std::clamp(std::stoi(argv[++i]),0,10000);
        else if(a=="--topk-probe-candidates"&&i+1<argc) topkProbeCandidates=std::clamp(std::stoi(argv[++i]),2,16);
        else if(a=="--topk-probe-ms"&&i+1<argc) topkProbeMs=std::clamp(std::stoi(argv[++i]),1,100);
        else if(a=="--topk-probe-depth"&&i+1<argc) topkProbeDepth=std::clamp(std::stoi(argv[++i]),1,16);
        else if(a=="--topk-min-ply"&&i+1<argc) topkMinPly=std::clamp(std::stoi(argv[++i]),0,225);
        else if(a=="--topk-max-ply"&&i+1<argc) topkMaxPly=std::clamp(std::stoi(argv[++i]),0,225);
        else if(a=="--midgame-start-prob"&&i+1<argc) midgameStartProb=std::clamp(std::stod(argv[++i]),0.0,1.0);
        else if(a=="--midgame-start-min-ply"&&i+1<argc) midgameStartMinPly=std::clamp(std::stoi(argv[++i]),4,100);
        else if(a=="--midgame-start-max-ply"&&i+1<argc) midgameStartMaxPly=std::clamp(std::stoi(argv[++i]),4,120);
        else if(a=="--midgame-seed-ms"&&i+1<argc) midgameSeedMs=std::clamp(std::stoi(argv[++i]),0,100);
        else if(a=="--midgame-seed-depth"&&i+1<argc) midgameSeedDepth=std::clamp(std::stoi(argv[++i]),1,16);
        else if(a=="--midgame-seed-static-topk"&&i+1<argc) midgameSeedStaticTopK=std::clamp(std::stoi(argv[++i]),1,16);
        else if(a=="--policy-record-topk"&&i+1<argc) policyRecordTopK=std::clamp(std::stoi(argv[++i]),1,4);
        else if(a=="--uncertainty-probe-prob"&&i+1<argc) uncertaintyProbeProb=std::clamp(std::stod(argv[++i]),0.0,1.0);
        else if(a=="--uncertainty-probe-ms"&&i+1<argc) uncertaintyProbeMs=std::clamp(std::stoi(argv[++i]),1,100);
        else if(a=="--uncertainty-probe-depth"&&i+1<argc) uncertaintyProbeDepth=std::clamp(std::stoi(argv[++i]),1,16);
        else if(a=="--short-target-probe-prob"&&i+1<argc) shortTargetProbeProb=std::clamp(std::stod(argv[++i]),0.0,1.0);
        else if(a=="--short-probe-ms"&&i+1<argc) shortProbeMs=std::clamp(std::stoi(argv[++i]),1,100);
        else if(a=="--short-probe-depth"&&i+1<argc) shortProbeDepth=std::clamp(std::stoi(argv[++i]),1,16);
        else if(a=="--seed"&&i+1<argc) seed=static_cast<uint32_t>(std::stoul(argv[++i]));
        else if(a=="--nnue"&&i+1<argc) nnuePath=argv[++i];
        else if(a=="--scalar-nnue") scalarNnue=true;
        else if(a=="--out"&&i+1<argc) outPath=argv[++i];
        else if(a=="--aux-out"&&i+1<argc) auxPath=argv[++i];
        else if(a=="--teacher-out"&&i+1<argc) teacherPath=argv[++i];
    }
    if(midgameStartMinPly>midgameStartMaxPly) std::swap(midgameStartMinPly,midgameStartMaxPly);
    if(topkMinPly>topkMaxPly) std::swap(topkMinPly,topkMaxPly);
    if(auxPath.empty()) auxPath=outPath+".aux";
    if(teacherPath.empty()) teacherPath=outPath+".teach";
    if(workers==0){ unsigned hw=std::max(1u,std::thread::hardware_concurrency()); workers=std::max(1u,hw/2); }
    workers=std::min(workers,games);

    if(!nnuePath.empty()) {
        Searcher probe(1); std::string err; probe.set_nnue_simd(!scalarNnue);
        if(!probe.load_nnue(nnuePath,&err)){std::cerr<<"cannot load NNUE: "<<err<<" path="<<nnuePath<<"\n";return 2;}
    }
    if(std::ifstream(outPath,std::ios::binary).good() || std::ifstream(auxPath,std::ios::binary).good() || std::ifstream(teacherPath,std::ios::binary).good()){
        std::cerr<<"refusing to overwrite existing output\n"; return 2;
    }
    std::ofstream out(outPath,std::ios::binary|std::ios::trunc), aux(auxPath,std::ios::binary|std::ios::trunc), teacher(teacherPath,std::ios::binary|std::ios::trunc);
    if(!out || !aux || !teacher){std::cerr<<"cannot open output/aux/teacher\n";return 2;}

    std::atomic<int> nextGame{0};
    std::atomic<uint64_t> written{0}, midgameGames{0}, randomizedSamples{0};
    std::mutex outMutex, logMutex;

    auto workerFn=[&](int workerId){
        std::mt19937 rng(seed + static_cast<uint32_t>(workerId)*0x9E3779B9u);
        Searcher searcher(static_cast<size_t>(hashMb)); searcher.set_threads(searchThreads); searcher.set_nnue_simd(!scalarNnue);
        if(!nnuePath.empty()){std::string err; if(!searcher.load_nnue(nnuePath,&err)){
            std::lock_guard<std::mutex> lock(logMutex); std::cerr<<"worker NNUE load failed: "<<err<<"\n"; return;
        }}
        while(true){
            int g=nextGame.fetch_add(1); if(g>=games) break;
            Position p; std::vector<PendingSample> samples; samples.reserve(120);
            Color winner=Color::Black; bool hasWinner=false;
            const bool useMidgame = std::generate_canonical<double,32>(rng) < midgameStartProb;
            int startPly=0;
            if(useMidgame){
                const int target=std::uniform_int_distribution<int>(midgameStartMinPly,midgameStartMaxPly)(rng);
                if(build_midgame_seed(p,searcher,rng,target,midgameSeedMs,midgameSeedDepth,midgameSeedStaticTopK)){
                    startPly=p.ply(); midgameGames.fetch_add(1);
                } else p.clear();
            }
            const int normalOpening = startPly>0 ? 0 : openingRandomPlies;
            for(int turn=p.ply();turn<225 && !game_won(p);turn++){
                auto lm=legal(p); if(lm.empty()) break;
                Move played, teacherBest; int teacherScore=0; bool randomized=false; Choice c;
                const bool searched=(startPly>0) || (turn>=normalOpening);
                if(!searched){
                    sort_static(p,lm);
                    const size_t cap=std::min<size_t>(lm.size(),6);
                    played=lm[std::uniform_int_distribution<size_t>(0,cap-1)(rng)]; teacherBest=played;
                } else {
                    c=searched_choice(searcher,p,rng,depth,movetime,topkRandomProb,topk,topkMargin,
                                           topkProbeCandidates,topkProbeMs,topkProbeDepth,topkMinPly,topkMaxPly,policyRecordTopK);
                    played=c.played; teacherBest=c.teacherBest; teacherScore=c.teacherScore; randomized=c.randomized;
                }
                if(!played.valid()) break;
                if(searched && p.ply()>=minRecordPly && p.ply()<=maxRecordPly && ((p.ply()-normalOpening)%recordStride==0)){
                    PendingSample ps; ps.stm=p.side_to_move();
                    for(int k=0;k<4;k++){ps.s.black[k]=p.stones(Color::Black).w[k];ps.s.white[k]=p.stones(Color::White).w[k];}
                    ps.s.searchScore=encode_teacher_score(teacherScore); ps.s.result=0; ps.s.side=static_cast<uint8_t>(p.side_to_move());
                    ps.s.ply=static_cast<uint16_t>(p.ply()); ps.s.bestSq=teacherBest.valid()?teacherBest.sq:played.sq;
                    for(int qi=0;qi<4;++qi){ps.t.topSq[qi]=0xFFFFu;ps.t.topScore[qi]=0;}
                    ps.t.topCount=static_cast<uint8_t>(std::clamp(c.topCount,0,4));
                    for(int qi=0;qi<c.topCount && qi<4;++qi){ps.t.topSq[qi]=c.topMoves[static_cast<size_t>(qi)].sq;ps.t.topScore[qi]=encode_teacher_score(c.topScores[static_cast<size_t>(qi)]);}
                    ps.t.depth=static_cast<uint8_t>(std::clamp(c.teacherDepth,0,255));
                    ps.t.nodes=static_cast<uint32_t>(std::min<uint64_t>(c.teacherNodes,0xFFFFFFFFULL));
                    if(c.topCount>=2) ps.t.margin=static_cast<int16_t>(std::clamp(c.topScores[0]-c.topScores[1],-32767,32767));
                    ps.t.short2=ps.t.short6=static_cast<int16_t>(-32768);
                    if(std::generate_canonical<double,32>(rng)<uncertaintyProbeProb){
                        SearchLimits ul; ul.maxDepth=uncertaintyProbeDepth; ul.moveTimeMs=uncertaintyProbeMs;
                        auto shallow=searcher.search(p,ul);
                        ps.t.uncertainty=static_cast<uint16_t>(std::clamp(std::abs(teacherScore-shallow.score),0,65535)); ps.t.flags|=1u;
                    }
                    if(std::generate_canonical<double,32>(rng)<shortTargetProbeProb){
                        int sv=0;
                        if(pv_future_probe(searcher,p,c.pv,2,shortProbeMs,shortProbeDepth,sv)){ps.t.short2=encode_teacher_score(sv);ps.t.flags|=2u;}
                        if(pv_future_probe(searcher,p,c.pv,6,shortProbeMs,shortProbeDepth,sv)){ps.t.short6=encode_teacher_score(sv);ps.t.flags|=4u;}
                    }
                    if(startPly>0) ps.flags|=1u; if(randomized){ps.flags|=2u; randomizedSamples.fetch_add(1);} samples.push_back(ps);
                }
                p.make_move(played);
                if(game_won(p)){winner=p.last_color();hasWinner=true;break;}
            }
            const uint16_t finalPly=static_cast<uint16_t>(std::clamp(p.ply(),0,225));
            for(auto &ps:samples) if(hasWinner) ps.s.result = ps.stm==winner ? 1 : -1;
            {
                std::lock_guard<std::mutex> lock(outMutex);
                for(const auto& ps:samples){
                    out.write(reinterpret_cast<const char*>(&ps.s),sizeof(ps.s));
                    AuxSample ax{finalPly,static_cast<uint8_t>(std::clamp(startPly,0,225)),ps.flags};
                    aux.write(reinterpret_cast<const char*>(&ax),sizeof(ax));
                    teacher.write(reinterpret_cast<const char*>(&ps.t),sizeof(ps.t));
                }
            }
            written.fetch_add(samples.size());
            {
                std::lock_guard<std::mutex> lock(logMutex);
                std::cerr<<"game "<<(g+1)<<"/"<<games<<" worker="<<workerId<<" samples="<<samples.size()
                         <<" result="<<(hasWinner?(winner==Color::Black?"B":"W"):"D")
                         <<" start="<<startPly<<" final="<<finalPly<<"\n";
            }
        }
    };

    std::vector<std::thread> pool; pool.reserve(workers);
    for(int w=0;w<workers;++w) pool.emplace_back(workerFn,w);
    for(auto& t:pool) t.join();
    out.flush(); aux.flush(); teacher.flush();
    std::cerr<<"written samples="<<written.load()<<" sample_bytes="<<sizeof(RawSample)<<" aux_bytes="<<sizeof(AuxSample)
             <<" workers="<<workers<<" search_threads="<<searchThreads<<" record_stride="<<recordStride
             <<" teacher_bytes="<<sizeof(TeacherSampleV2)
             <<" eval="<<(nnuePath.empty()?"classical":(scalarNnue?"nnue-scalar":"nnue-auto"))
             <<" midgame_games="<<midgameGames.load()<<" randomized_samples="<<randomizedSamples.load()<<"\n";
    return (out && aux && teacher) ? 0 : 3;
}
