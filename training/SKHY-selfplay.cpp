#include <algorithm>
#include <atomic>
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
#pragma pack(pop)
static_assert(sizeof(RawSample)==72, "training sample format changed");

struct PendingSample { RawSample s{}; Color stm=Color::Black; };

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

int main(int argc,char**argv){
    int games=1, depth=5, movetime=100, workers=0, hashMb=32, searchThreads=1, openingRandomPlies=4, recordStride=1, minRecordPly=0, maxRecordPly=225;
    uint32_t seed=0x1A5B7C9Du;
    std::string outPath="data/selfplay_00000.bin";
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
        else if(a=="--seed"&&i+1<argc) seed=static_cast<uint32_t>(std::stoul(argv[++i]));
        else if(a=="--nnue"&&i+1<argc) nnuePath=argv[++i];
        else if(a=="--scalar-nnue") scalarNnue=true;
        else if(a=="--out"&&i+1<argc) outPath=argv[++i];
    }
    if(workers==0){
        unsigned hw=std::max(1u,std::thread::hardware_concurrency());
        workers=std::max(1u,hw/2);
    }
    workers=std::min(workers,games);

    if(!nnuePath.empty()) {
        Searcher probe(1); std::string err; probe.set_nnue_simd(!scalarNnue);
        if(!probe.load_nnue(nnuePath,&err)){std::cerr<<"cannot load NNUE: "<<err<<" path="<<nnuePath<<"\n";return 2;}
    }

    if(std::ifstream(outPath,std::ios::binary).good()){std::cerr<<"refusing to overwrite existing shard: "<<outPath<<"\n";return 2;}
    std::ofstream out(outPath,std::ios::binary|std::ios::trunc);
    if(!out){std::cerr<<"cannot open output: "<<outPath<<"\n";return 2;}

    std::atomic<int> nextGame{0};
    std::atomic<uint64_t> written{0};
    std::mutex outMutex, logMutex;

    auto workerFn=[&](int workerId){
        std::mt19937 rng(seed + static_cast<uint32_t>(workerId)*0x9E3779B9u);
        Searcher searcher(static_cast<size_t>(hashMb));
        searcher.set_threads(searchThreads);
        searcher.set_nnue_simd(!scalarNnue);
        if(!nnuePath.empty()){std::string err; if(!searcher.load_nnue(nnuePath,&err)){
            std::lock_guard<std::mutex> lock(logMutex); std::cerr<<"worker NNUE load failed: "<<err<<"\n"; return;
        }}
        while(true){
            int g=nextGame.fetch_add(1);
            if(g>=games) break;
            Position p; std::vector<PendingSample> samples; samples.reserve(120);
            Color winner=Color::Black; bool hasWinner=false;
            for(int turn=0;turn<225 && !game_won(p);turn++){
                auto lm=legal(p); if(lm.empty()) break;
                Move chosen; int score=0;
                const bool searched=turn>=openingRandomPlies;
                if(!searched){
                    if(lm.size()==1) chosen=lm.front();
                    else {
                        std::sort(lm.begin(),lm.end(),[](Move a,Move b){
                            int da=std::abs(a.x()-7)+std::abs(a.y()-7), db=std::abs(b.x()-7)+std::abs(b.y()-7); return da<db;
                        });
                        size_t cap=std::min<size_t>(lm.size(),6); chosen=lm[std::uniform_int_distribution<size_t>(0,cap-1)(rng)];
                    }
                } else {
                    SearchLimits lim; lim.maxDepth=depth; lim.moveTimeMs=movetime;
                    auto r=searcher.search(p,lim); chosen=r.bestMove; score=r.score;
                }
                if(!chosen.valid()) break;
                if(searched && p.ply()>=minRecordPly && p.ply()<=maxRecordPly && ((p.ply()-openingRandomPlies)%recordStride==0)){
                    PendingSample ps; ps.stm=p.side_to_move();
                    for(int k=0;k<4;k++){ps.s.black[k]=p.stones(Color::Black).w[k];ps.s.white[k]=p.stones(Color::White).w[k];}
                    ps.s.searchScore=static_cast<int16_t>(std::clamp(score,-10000,10000)); ps.s.result=0;
                    ps.s.side=static_cast<uint8_t>(p.side_to_move()); ps.s.ply=static_cast<uint16_t>(p.ply()); ps.s.bestSq=chosen.sq;
                    samples.push_back(ps);
                }
                p.make_move(chosen);
                if(game_won(p)){winner=p.last_color();hasWinner=true;break;}
            }
            for(auto &ps:samples) if(hasWinner) ps.s.result = ps.stm==winner ? 1 : -1;
            {
                std::lock_guard<std::mutex> lock(outMutex);
                for(const auto& ps:samples) out.write(reinterpret_cast<const char*>(&ps.s),sizeof(ps.s));
            }
            written.fetch_add(samples.size());
            {
                std::lock_guard<std::mutex> lock(logMutex);
                std::cerr<<"game "<<(g+1)<<"/"<<games<<" worker="<<workerId<<" samples="<<samples.size()
                         <<" result="<<(hasWinner?(winner==Color::Black?"B":"W"):"D")<<"\n";
            }
        }
    };

    std::vector<std::thread> pool; pool.reserve(workers);
    for(int w=0;w<workers;++w) pool.emplace_back(workerFn,w);
    for(auto& t:pool) t.join();
    out.flush();
    std::cerr<<"written samples="<<written.load()<<" sample_bytes="<<sizeof(RawSample)<<" workers="<<workers
             <<" search_threads="<<searchThreads<<" record_stride="<<recordStride<<" eval="<<(nnuePath.empty()?"classical":(scalarNnue?"nnue-scalar":"nnue-auto"))<<"\n";
    return out ? 0 : 3;
}
