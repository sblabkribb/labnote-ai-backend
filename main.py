# labnote-ai-backend/main.py

import os
import logging
import datetime
import uuid
import re
import asyncio
import json
import redis.asyncio as redis
import sqlite3, datetime
import ollama
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
from typing import Optional, List, Dict
from contextlib import asynccontextmanager
from dotenv import load_dotenv
from pathlib import Path
from rapidfuzz import fuzz
import git
from fastapi.middleware.cors import CORSMiddleware 

# Local imports
import rag_pipeline as rag_module
from agents import run_agent_team
from llm_utils import call_llm_api

# embedding
from rag_pipeline import get_embeddings

# .env 파일 로드 및 로깅 설정
load_dotenv()
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- 데이터 사전 처리 (기존과 동일) ---
WORKFLOW_GUIDE_DATA = """
# Workflows Guide
## Design (설계)
- WD010: General Design of Experiment (실험설계법(DOE)을 활용한 범용적 실험 조건 최적화)
- WD020: Adaptive Laboratory Evolution Design (무작위 돌연변이 및 인공 진화를 통한 하향식 설계)
- WD030: Growth Media Design (데이터 기반 실험 설계를 통한 균주 배양용 성장 배지 최적화)
- WD040: Parallel Cell Culture/Fermentation Design (단백질, 효소 대량 배양 또는 균주 활성 테스트 조건 설계)
- WD050: DNA Oligomer Pool Design (목표 DNA 서열 조립을 위한 올리고머 풀 설계)
- WD060: Genetic Circuit Design (바이오센서, 논리 게이트 등 특정 목적의 유전 회로 설계)
- WD070: Vector Design (플라스미드, BAC, YAC 등 벡터 형태의 DNA 구축 설계)
- WD080: Artificial Genome Design (유전체 압축, 코돈 재설계 등 새로운 유전체 디자인)
- WD090: Genome Editing Design (CRISPR 기반 유전체 편집을 위한 gRNA 설계)
- WD100: Protein Library Design (단백질 활성, 특이성, 발현 최적화를 위한 라이브러리 설계)
- WD110: De novo Protein/Enzyme Design (딥러닝 도구를 이용한 새로운 단백질 또는 효소 설계)
- WD120: Retrosynthetic Pathway Design (역합성 분석을 통한 목표 대사산물 생산 경로 설계)
- WD130: Pathway Library Design (대사 경로 기능 최적화를 위한 DNA 부품 라이브러리 설계)
## Build (구축)
- WB005: Nucleotide Quantification (UV 흡광도 및 형광 분석을 통한 핵산 정량 및 순도 평가)
- WB010: DNA Oligomer Assembly (DNA 올리고머 풀로부터 정확한 DNA 서열 조립)
- WB020: DNA Library Construction (DNA 돌연변이, 메타게놈, 경로 라이브러리 제작)
- WB025: Sequencing Library Preparation (차세대 시퀀싱(NGS)을 위한 DNA/RNA 라이브러리 준비)
- WB030: DNA Assembly (여러 DNA 단편을 특정 순서로 조립하여 유전 구조물 제작)
- WB040: DNA Purification (컬럼, 비드 등을 이용해 조 DNA 추출물에서 고순도 DNA 정제)
- WB045: DNA Extraction (세포 용해를 통해 생물학적 샘플로부터 DNA 추출)
- WB050: RNA Extraction (유전자 발현 분석 등을 위해 생물학적 샘플에서 RNA 분리)
- WB060: DNA Multiplexing (식별을 위해 세포에 바코드를 할당하고 NGS용 DNA 풀링)
- WB070: Cell-free Mixture Preparation (무세포 반응을 위한 마스터 용액 및 세포 추출물 준비)
- WB080: Cell-free Protein/Enzyme Expression (무세포 반응 시스템에서 목표 단백질 또는 효소 생산)
- WB090: Protein Purification (자동화 장비를 이용한 고처리량, 고순도 단백질 정제)
- WB100: Growth Media Preparation and Sterilization (설계된 고체 및 액체 배지의 대량 생산, 멸균 및 보관)
- WB110: Competent Cell Construction (형질전환을 위한 고효율의 Competent cell 제작)
- WB120: Biology-mediated DNA Transfers (설계된 벡터 플라스미드를 세포 내로 자동 형질전환)
- WB125: Colony Picking (자동화 콜로니 피커를 이용한 단일 콜로니 분리 및 배양)
- WB130: Solid Media Cell Culture (고체 배지에서의 세포 배양, 스크리닝 및 단일 콜로니 분리)
- WB140: Liquid Media Cell Culture (액체 배지에서의 접종 및 회분 배양 프로세스)
- WB150: PCR-based Target Amplification (PCR을 이용해 복잡한 템플릿에서 특정 유전자 서열 증폭)
## Test (시험)
- WT010: Nucleotide Sequencing (NGS 또는 Sanger 시퀀싱을 이용한 염기서열 데이터 생성)
- WT012: Targeted mRNA Expression Measurement (RT-qPCR, ddPCR 등을 이용한 특정 전사체 수준 측정)
- WT015: Nucleic Acid Size Verification (전기영동을 이용한 DNA/RNA 단편 크기 및 무결성 확인)
- WT020: Protein Expression Measurement (겔 전기영동, LC-MS 등을 통한 목표 단백질 발현 수준 정량화)
- WT030: Protein/Enzyme Activity Measurement (정제된 단백질 또는 효소의 활성을 특정 방법으로 측정)
- WT040: Parallel Cell-free Protein/Enzyme Reaction (무세포 시스템에서 단백질 발현과 활성을 동시에 측정)
- WT045: Mammalian Cell Cytotoxicity Assay (포유류/진핵 세포의 생존력 및 세포 독성 효과 정량화)
- WT046: Microbial Viability and Cytotoxicity Assay (미생물 세포의 성장 억제 및 생존력 측정 (MIC/MBC 등))
- WT050: Sample Pretreatment (배양액에서 대사체 분리 및 분석을 위한 전처리)
- WT060: Metabolite Measurement (GC-MS, LC-MS 등을 이용한 대사체 정량 분석)
- WT070: High-throughput Single Metabolite Measurement (바이오센서 등을 이용한 단일 유형 대사산물 고속 측정)
- WT080: Image Analysis (고속 광학 장치를 이용한 세포 성장, 형태, 위치 분석)
- WT085: Mycoplasma Contamination Test (포유류 세포 배양의 마이코플라즈마 오염 스크리닝)
- WT090: High-speed Cell Sorting (유전 회로 신호를 기반으로 특정 세포 집단 고속 분리)
- WT100: Micro-scale Parallel Cell Culture (96-딥웰 플레이트에서의 마이크로 스케일 병렬 세포 배양)
- WT110: Micro-scale Parallel Cell Fermentation (OD, pH, 온도, DO 모니터링을 통한 마이크로 스케일 발효)
- WT120: Parallel Cell Fermentation (15-250ml 규모의 실시간 모니터링 병렬 세포 발효)
- WT130: Parallel Mammalian Cell Fermentation (단백질 생산 극대화를 위한 동물 세포 병렬 발효)
- WT140: Lab-scale Fermentation (10L 미만 규모의 실험실 스케일 발효 공정 개발)
- WT150: Pilot-scale Fermentation (10L-500L 규모의 파일럿 스케일 발효 공정)
- WT160: Industrial-scale Fermentation (500L 이상 산업 스케일의 대규모 발효 공정)
## Learn (학습)
- WL010: Sequence Variant Analysis (유전자, 플라스미드 등 주형 DNA 서열의 변이 비교 분석)
- WL020: Genome Resequencing Analysis (참조 유전체가 있는 생물체의 SNP 등 유전체 변이 분석)
- WL030: De novo Genome Analysis (참조 유전체가 없는 신규 생물체의 유전체 조립 및 분석)
- WL040: Metagenomic Analysis (대용량 메타게놈 서열 데이터의 유전자 및 균주 식별, 기능 예측)
- WL050: Transcriptome Analysis (다양한 조건 하의 전사체(mRNA) 데이터 및 유전자 발현 차이 분석)
- WL055: Single Cell Analysis (단일 세포 RNA 시퀀싱 등을 통한 세포 이질성 및 기능 분석)
- WL060: Metabolic Pathway Optimization Model Development (측정된 대사체 데이터 분석 및 대사 경로 최적화 모델 개발)
- WL070: Phenotypic Data Analysis (표현형 데이터 처리 및 분석을 통한 유전형-표현형 관계 규명)
- WL080: Protein/Enzyme Optimization Model Development (단백질/효소의 특성(활성, 용해도 등) 최적화 모델 개발)
- WL090: Fermentation Optimization Model Development (발효 데이터를 기반으로 목표 화합물 생산 최적 조건 탐색)
- WL100: Foundation Model Development (대규모 서열 데이터셋을 이용한 파운데이션 모델 훈련)
"""
UNIT_OPERATION_GUIDE_DATA = """
# Unit Operations Guide
## Hardware (UHW)
- UHW010: Liquid Handling (액체 시약의 정밀 분주, 희석, 혼합 등 기본 작업)
- UHW015: Bulk Liquid Dispenser (배지, 버퍼 등 대용량 액체의 빠른 분배)
- UHW020: 96 Channel Liquid Handling (96-웰 플랫폼에서의 고처리량 동시 액체 분주/전송)
- UHW030: Nanoliter Liquid Dispensing (나노리터 단위의 초미세 액체 정밀 분주)
- UHW040: Desktop Liquid Handling (소규모 자동화 실험을 위한 소형 액체 핸들링 시스템)
- UHW050: Single Cell Sequencing Preparation (단일 세포 분석을 위한 세포 캡슐화 및 라이브러리 준비)
- UHW060: Colony Picking (한천 배지에서 단일 콜로니를 분리하여 액체 배양)
- UHW070: Cell Sorting (세포의 생물학적 특성에 따른 고속 세포 분류 및 선택)
- UHW080: Cell Lysis (세포를 파괴하여 내부 구성물(DNA, 단백질 등) 추출)
- UHW090: Electroporation (전기장을 이용해 세포 내로 DNA, RNA 등 외부 분자 도입)
- UHW100: Thermocycling (PCR 등 반응 촉진을 위한 반복적인 온도 순환)
- UHW110: Real-time PCR (특정 DNA/RNA 서열의 증폭 및 실시간 정량 분석)
- UHW120: Plate Handling (로봇 팔을 이용한 자동화 장비 간 플레이트 이동)
- UHW130: Sealing (PCR, 배양, 보관 시 샘플 무결성을 위한 플레이트 밀봉)
- UHW140: Peeling (자동화 공정을 위한 플레이트 덮개 제거)
- UHW150: Capping Decapping (샘플 튜브 캡의 자동 개폐)
- UHW160: Sample Storage (자동화된 DNA 또는 세포 샘플 저장 및 검색 시스템)
- UHW170: Plate Storage (고처리량 실험을 위한 자동화 플레이트 저장 및 검색)
- UHW180: Incubation (세포 성장 및 반응을 위한 특정 조건 유지 (온도, 습도 등))
- UHW190: HT Aerobic Fermentation (산소 조건에서의 고처리량 병렬 미생물/세포 배양)
- UHW200: HT Anaerobic Fermentation (무산소 조건에서의 고처리량 병렬 미생물/세포 배양)
- UHW210: Microbioreactor Fermentation (고급 모니터링 기능의 마이크로 규모 생물반응기 배양)
- UHW220: Bioreactor Fermentation (리터 규모 생물반응기에서의 세포 배양 (회분, 유가, 연속))
- UHW230: Nucleic Acid Fragment Analysis (크기 기반 핵산 단편 분리, 식별 및 특성 분석)
- UHW240: Protein Fragment Analysis (단백질 단편의 구조, 크기, 변형, 상호작용 연구)
- UHW250: Nucleic Acid Purification (자동화 장치를 이용한 고순도 DNA/RNA 정제)
- UHW255: Centrifuge (원심력을 이용한 샘플 내 밀도 별 성분 분리)
- UHW260: Short-read Sequence Analysis (NGS 기술을 이용한 짧은 서열 기반 시퀀싱)
- UHW265: Sanger Sequencing (표적 유전자/플라스미드 검증을 위한 전통적 시퀀싱)
- UHW270: Long-read Sequence Analysis (복잡한 유전체 영역 분석을 위한 긴 서열 기반 시퀀싱)
- UHW280: Sequence Quality Control (단일 세포 분석을 위한 시퀀싱 데이터 품질 평가)
- UHW290: LC-MS-MS (탠덤 질량분석기가 결합된 고성능 액체 크로마토그래피)
- UHW300: LC-MS (질량분석기가 결합된 액체 크로마토그래피)
- UHW310: HPLC (고성능 액체 크로마토그래피)
- UHW320: UPLC (초고성능 액체 크로마토그래피)
- UHW330: GC (가스 크로마토그래피)
- UHW340: GC-MS (질량분석기가 결합된 가스 크로마토그래피)
- UHW350: GC-MS-MS (탠덤 질량분석기가 결합된 가스 크로마토그래피)
- UHW355: SPE-MS-MS (고체상 추출 및 탠덤 질량 분석)
- UHW360: FPLC (단백질 등 생체 분자 정제에 최적화된 고속 단백질 액체 크로마토그래피)
- UHW365: Rapid Sugar Analyzer (효소 기반 센서를 이용한 특정 당(포도당 등)의 신속 정량)
- UHW370: Oligomer Synthesis (화학적 방법을 이용한 맞춤형 DNA/RNA 올리고머 병렬 합성)
- UHW380: Microplate Reading (형광, OD 등을 측정하여 단백질/세포 활성 정량화)
- UHW390: Microscopy Imaging (동물 세포 등 생물학적 샘플의 현미경 이미지 촬영)
- UHW400: Manual (시약 준비, 실험기구 준비 등 수동으로 진행되는 모든 실험 과정)
## Software (USW)
- USW005: Biological Database (표준 생물학적 부품 데이터베이스 검색 및 선택)
- USW010: DNA Oligomer Pool Design (효율적인 DNA 조립을 위한 올리고머 풀 설계)
- USW020: Primer Design (PCR, 돌연변이 생성 등을 위한 프라이머 설계)
- USW030: Vector Design (삽입 서열과 플라스미드 백본을 고려한 벡터 맵 설계)
- USW040: Sequence Optimization (특정 숙주에서 단백질 발현을 극대화하기 위한 코돈 최적화)
- USW050: Synthesis Screening (생물 보안을 위한 잠재적 위험 DNA 서열 스크리닝)
- USW060: Structure-based Sequence Generation (AI 모델을 이용한 단백질 구조 기반 서열 생성)
- USW070: Protein Structure Prediction (AI 모델을 이용한 단백질 3차 구조 예측)
- USW080: Protein Structure Generation (AI 모델을 이용한 새로운 기능의 단백질 구조 생성)
- USW090: Retrosynthetic Pathway Design (역합성 분석을 통한 생합성 경로 예측 및 신규 경로 발견)
- USW100: Enzyme Identification (데이터베이스 검색 또는 예측을 통한 경로 내 적합 효소 탐색)
- USW110: Sequence Alignment (서열 유사성 비교 및 상동 서열 식별)
- USW120: Sequence Trimming and Filtering (데이터 품질 향상을 위한 저품질 시퀀싱 리드 제거)
- USW130: Read Mapping and Alignment (시퀀싱 리드를 참조 서열에 매핑 및 정렬)
- USW140: Sequence Assembly (시퀀싱 리드를 조립하여 전체 유전자, 경로, 염색체 재구성)
- USW145: Metagenomic Assembly (복잡한 미생물 군집으로부터 유전체 재구성)
- USW150: Sequence Quality Control (FastQ, Fast5 등 시퀀싱 파일 품질 관리(QC))
- USW160: Demultiplexing (바코드 기반으로 NGS 리드를 개별 샘플로 분리)
- USW170: Variant Calling (리드 매핑 기반의 SNP, indel 등 변이 탐지)
- USW180: RNA-Seq Analysis (전사체 데이터 처리 및 유전자 발현 정량화 분석)
- USW185: Gene Set Enrichment Analysis (유전자 발현 데이터에서 유의미한 생물학적 경로 분석)
- USW190: Proteomics Data Analysis (질량 분석 데이터 처리 및 단백질 식별/정량 분석)
- USW200: Phylogenetic Analysis (서열 유사성에 기반한 계통 발생 관계 분석)
- USW210: Metabolic Flux Analysis (세포 대사 및 경로 최적화를 위한 대사 흐름 모델링/분석)
- USW220: Deep Learning Data Preparation (AI 모델 훈련 및 평가를 위한 데이터셋 준비 및 배치화)
- USW230: Sequence Embedding (생물학적 서열을 기계 학습용 수치 표현으로 변환)
- USW240: Deep Learning Model Training (훈련 데이터를 이용한 딥러닝 모델 훈련 절차)
- USW250: Model Evaluation (정확도, 정밀도 등 평가지표를 이용한 모델 성능 평가)
- USW260: Hyperparameter Tuning (베이즈 최적화 등을 이용한 모델 하이퍼파라미터 튜닝)
- USW270: Model Deployment (훈련된 모델을 서비스로 배포)
- USW280: Monitoring and Reporting (배포된 AI 모델의 성능 및 자원 사용량 모니터링)
- USW290: Phenotype Data Preprocessing (측정된 표현형 데이터의 정제, 구성, 변환 등 전처리)
- USW300: XCMS Analysis (크로마토그래피 및 질량분석 데이터 분석 및 시각화)
- USW310: Flow Cytometry Analysis (유세포 분석 데이터 분석 및 시각화)
- USW320: DNA Assembly Simulation (Golden Gate, Gibson 등 DNA 조립 성공률 향상을 위한 시뮬레이션)
- USW325: Gene Editing Simulation (CRISPR 유전자 편집 결과 및 표적 이탈 효과 예측 시뮬레이션)
- USW330: Well Plate Mapping (고처리량 스크리닝을 위한 웰 플레이트 매핑 소프트웨어)
- USW340: Computation (일반적인 데이터 수집, 전처리, 분석 과정)
"""

def _precompute_data():
    logger.info("Pre-computing static data (ALL_UOS, ALL_WORKFLOWS)...")
    all_uos = {m.group(1): m.group(2).strip() for m in re.finditer(r'- ([A-Z]{2,3}\d{3}): (.*)', UNIT_OPERATION_GUIDE_DATA)}
    all_workflows = {m.group(1): m.group(2).strip() for m in re.finditer(r'- ([A-Z]{2}\d{3}): (.*)', WORKFLOW_GUIDE_DATA)}
    logger.info(f"Loaded {len(all_workflows)} workflows and {len(all_uos)} unit operations.")
    return all_uos, all_workflows

ALL_UOS_DATA, ALL_WORKFLOWS_DATA = _precompute_data()

# --- Redis 연결 관리 (RAG 파이프라인 전용) ---
redis_pool = None

async def keep_gpu_warm():
    """5분(300초)마다 임베딩 연산을 수행하여 GPU를 활성 상태로 유지합니다."""
    while True:
        # ⭐️ 개선점: 가장 크고 중요한 llama3:70b 모델을 메모리에 유지하도록 변경합니다.
        # 이렇게 하면 모델을 계속해서 로드/언로드하는 것을 방지할 수 있습니다.
        try:
            logger.info("[Keep-Alive] Running scheduled GPU health check...")
            await ollama.AsyncClient().chat(
                model='llama3:70b',
                messages=[{'role': 'user', 'content': 'Health check. Respond with "OK".'}],
                options={'num_predict': 1} # 최소한의 작업만 수행
            )
            logger.info("[Keep-Alive] Successfully kept llama3:70b model warm.")
        except Exception as e:
            logger.error(f"[Keep-Alive] Error during GPU health check: {e}", exc_info=True)
        
        await asyncio.sleep(300)

@asynccontextmanager
async def lifespan(app: FastAPI):
    global redis_pool
    redis_url = os.getenv("REDIS_URL")
    if not redis_url:
        raise ValueError("REDIS_URL environment variable is not set.")
    logger.info(f"Creating Redis connection pool for {redis_url}")
    
    # ⭐️ [수정] 서버 시작 시 RAG 파이프라인을 초기화합니다.
    logger.info("Initializing RAG pipeline...")
    rag_module.rag_pipeline = rag_module.RAGPipeline()
    
    redis_pool = redis.ConnectionPool.from_url(redis_url, decode_responses=True)
    logger.info("Starting background task to keep GPU warm...")
    asyncio.create_task(keep_gpu_warm())
    yield
    logger.info("Closing Redis connection pool.")
    await redis_pool.disconnect()

# FastAPI 앱 초기화
app = FastAPI(
    title="LabNote AI Assistant Backend",
    version="2.5.0",
    description="Interactive lab note generation with user-edit DPO feedback loop and consent management.",
    lifespan=lifespan
)

# 출처 허용
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 모든 출처 허용
    allow_credentials=True,
    allow_methods=["*"],  # 모든 HTTP 메소드 허용
    allow_headers=["*"],  # 모든 HTTP 헤더 허용
)

# --- 템플릿 설정 ---
templates_dir = Path(__file__).parent / "templates"
templates = Jinja2Templates(directory=str(templates_dir))

# --- Pydantic 모델 정의 ---
conversation_histories: Dict[str, List[Dict[str, str]]] = {}

class CreateScaffoldRequest(BaseModel):
    query: str
    workflow_id: str
    unit_operation_ids: List[str]
    experimenter: Optional[str] = "AI Assistant"

class LabNoteResponse(BaseModel):
    files: Dict[str, str]

class PopulateNoteRequest(BaseModel):
    file_content: str
    uo_id: str
    section: str
    query: str

class PopulateNoteResponse(BaseModel):
    uo_id: str
    section: str
    options: List[str]

class GitFeedbackRequest(BaseModel):
    prompt: str
    chosen: str
    rejected: List[str]
    metadata: Dict

class PreferenceRequest(BaseModel):
    uo_id: str
    section: str
    chosen_original: str
    chosen_edited: str
    rejected: List[str]
    query: str
    file_content: str
    file_path: str
    supervisor_evaluations: List[Dict]

class ChatRequest(BaseModel):
    query: str
    conversation_id: Optional[str] = None

class ChatResponse(BaseModel):
    response: str
    conversation_id: str

# --- 헬퍼 함수 ---
def get_seoul_date_string():
    return datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=9))).strftime('%Y-%m-%d')

def create_unit_operation_template(uo_id: str, uo_name: str, experimenter: str) -> str:
    formatted_datetime = datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=9))).strftime('%Y-%m-%d %H:%M')
    return f"""
### [{uo_id} {uo_name}]

#### Meta
- Experimenter: {experimenter}
- Start_date: '{formatted_datetime}'
- End_date: ''

#### Input
- (samples from the previous step)

#### Reagent
- (e.g. enzyme, buffer, etc.)

#### Consumables
- (e.g. filter, well-plate, etc.)

#### Equipment
- (e.g. centrifuge, spectrophotometer, etc.)

#### Method
- (method used in this step)

#### Output
- (samples to the next step)

#### Results & Discussions
- (Any results and discussions. Link file path if needed)
"""

def _extract_section_content(uo_block: str, section_name: str) -> str:
    pattern = re.compile(r"#### " + re.escape(section_name) + r"\n(.*?)(?=\n####|\Z)", re.DOTALL)
    match = pattern.search(uo_block)
    if match:
        content = match.group(1).strip()
        return content if content and not content.startswith('(') else "(not specified)"
    return "(not specified)"

def _init_feedback_db():
    """
    피드백 지표를 저장하기 위한 SQLite 데이터베이스와 테이블을 초기화합니다.
    이 함수는 서버 시작 시 또는 첫 피드백 기록 시 호출될 수 있습니다.
    """
    db_path = os.getenv("EVALUATION_DB_PATH", "scripts/evaluation_results.db")
    # scripts 디렉토리가 없으면 생성
    os.makedirs(os.path.dirname(db_path), exist_ok=True)
    with sqlite3.connect(db_path) as conn:
        cursor = conn.cursor()
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS feedback_metrics (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                uo_id TEXT NOT NULL,
                section TEXT NOT NULL,
                edit_distance_ratio REAL NOT NULL
            )
        """)
        logger.info(f"Feedback metrics table initialized in '{db_path}'")


# --- API 엔드포인트 ---

@app.post("/create_scaffold", response_model=LabNoteResponse)
async def create_scaffold(request: CreateScaffoldRequest):
    logger.info(f"Corrected multi-file scaffold generation for WF: {request.workflow_id}")
    try:
        experimenter = request.experimenter
        formatted_date = get_seoul_date_string()
        
        wf_id = request.workflow_id
        wf_name = ALL_WORKFLOWS_DATA.get(wf_id, "Custom Workflow")
        
        wf_description = "> 이 워크플로의 설명을 간략하게 작성합니다 (아래 설명은 템플릿으로 사용자 목적에 맞도록 수정합니다)"
        
        workflow_file_name = f"001_{wf_id}_{wf_name.replace(' ', '_')}.md"

        unit_operation_blocks = []
        for uo_id in request.unit_operation_ids:
            uo_name = ALL_UOS_DATA.get(uo_id, "Unknown Operation")
            unit_operation_blocks.append(create_unit_operation_template(uo_id, uo_name, experimenter))
        
        all_uo_blocks_content = "\n\n".join(unit_operation_blocks)

        workflow_content = f"""---
title: "{wf_id} {wf_name}"
experimenter: "{experimenter}"
created_date: '{formatted_date}'
last_updated_date: '{formatted_date}'
---

## [{wf_id} {wf_name}]
{wf_description}

## 🗂️ 관련 유닛오퍼레이션

{all_uo_blocks_content}
"""
        link_text = f"001 {wf_id} {wf_name}"
        workflow_link = f"[ ] [{link_text}](./{workflow_file_name})"

        readme_content = f"""---
title: "{request.query}"
experimenter: "{experimenter}"
created_date: '{formatted_date}'
last_updated_date: '{formatted_date}'
experiment_type: labnote
---

## 🎯 실험 목표
> 이 실험의 주된 목표와 가설을 간략하게 작성합니다.

## 🗂️ 관련 워크플로
> 아래 표시 사이에 관련된 워크플로 파일 목록을 입력합니다.
> `F1`, `New workflow` 명령 수행시 해당 목록은 표시된 위치 사이에 자동 추가됩니다.

{workflow_link}
"""
        
        files_to_create = {
            "README.md": readme_content,
            workflow_file_name: workflow_content
        }

        return LabNoteResponse(files=files_to_create)

    except Exception as e:
        logger.error(f"Error during multi-file scaffold creation: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error creating scaffold: {e}")

@app.post("/populate_note", response_model=PopulateNoteResponse)
async def populate_note(request: PopulateNoteRequest):
    logger.info(f"Phase 2: Populating section '{request.section}' for UO '{request.uo_id}'")
    try:
        pattern = re.compile(
            r"(### \\?\[" + re.escape(request.uo_id) + r".*?\\?\]\n.*?)(?=### \\?\[U[A-Z]{2,3}\d{3}|\Z)",
            re.DOTALL
        )
        match = pattern.search(request.file_content)
        if not match:
            logger.error(f"Could not find UO block for ID '{request.uo_id}'. Searched content snippet: \n---\n{request.file_content[:500]}\n---")
            raise HTTPException(status_code=404, detail=f"Unit Operation block for ID '{request.uo_id}' not found.")
        
        uo_block = match.group(1)
        agent_result = await asyncio.to_thread(run_agent_team, request.query, uo_block, request.section)
        
        if not agent_result or not agent_result.get("options"):
            raise HTTPException(status_code=500, detail="Agent team failed to generate options.")
        
        return PopulateNoteResponse(**agent_result)
    except Exception as e:
        logger.error(f"Error populating note: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error populating note: {e}")

# Git 작업을 처리할 새로운 동기 함수
def _run_git_operations(token: str, repo_url: str, local_path_str: str, preference_data: dict, commit_message: str):
    """Git 작업을 동기적으로 처리하는 헬퍼 함수"""
    local_path = Path(local_path_str)
    repo_url_with_token = repo_url.replace("https://", f"https://oauth2:{token}@")

    if local_path.exists():
        repo = git.Repo(local_path)
        # git pull 시 rebase 하도록 설정 추가
        with repo.config_writer() as git_config:
            git_config.set_value('pull', 'rebase', 'true')
        logger.info("Pulling latest changes from DPO repository...")
        repo.remotes.origin.pull()
    else:
        logger.info(f"Cloning DPO repository to {local_path}...")
        repo = git.Repo.clone_from(repo_url_with_token, local_path)
        # 새로 클론한 저장소에도 rebase 설정 추가
        with repo.config_writer() as git_config:
            git_config.set_value('pull', 'rebase', 'true')
    data_dir = local_path / "data"
    data_dir.mkdir(exist_ok=True)
    
    file_name = f"{datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4()}.json"
    file_path = data_dir / file_name
    
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(preference_data, f, ensure_ascii=False, indent=2)
    
    logger.info(f"DPO data saved to {file_path}")

    repo.index.add([str(file_path.resolve())])
    repo.index.commit(commit_message)
    
    logger.info("Pushing DPO data to remote repository...")
    origin = repo.remote(name='origin')
    origin.push()
    logger.info("Successfully pushed DPO data to Git.")

@app.post("/record_preference", status_code=204)
async def record_preference(request: PreferenceRequest):
    logger.info(f"Recording DPO data for UO '{request.uo_id}' to Git repository.")

    repo_url = os.getenv("DPO_TRAINER_REPO_URL")
    token = os.getenv("GIT_AUTH_TOKEN")
    local_path_str = os.getenv("DPO_REPO_LOCAL_PATH", "./labnote-dpo-trainer-data")
    
    # 디버깅 로그 추가
    logger.info(f"DEBUG: Attempting to use GIT_AUTH_TOKEN: '{token[:4]}...{token[-4:] if token else 'None'}'")

    if not repo_url or not token:
        logger.error("Git repository URL or auth token is not configured in .env file.")
        raise HTTPException(status_code=500, detail="DPO Git repository is not configured on the server.")

    try:
        # preference_data 생성 로직 (기존과 동일)
        uo_name = ALL_UOS_DATA.get(request.uo_id, "Unknown Operation")
        uo_block_pattern = re.compile(r"(### \\?\[" + re.escape(request.uo_id) + r".*?\\?\]\n.*?)(?=### \\?\[U[A-Z]{2,3}\d{3}|\Z)", re.DOTALL)
        uo_match = uo_block_pattern.search(request.file_content)
        uo_block_content = uo_match.group(1) if uo_match else ""
        input_context = _extract_section_content(uo_block_content, "Input")
        output_context = _extract_section_content(uo_block_content, "Output")
        
        prompt = (
            f"Given the experimental context, write the '{request.section}' section for the Unit Operation '{request.uo_id}: {uo_name}'.\n"
            f"- Overall Goal: {request.query}\n"
            f"- Starting Materials (Input): {input_context}\n"
            f"- Desired End-Product (Output): {output_context}\n"
            f"- The initial AI suggestion was: {request.chosen_original}"
        )

        path_parts = request.file_path.replace("\\", "/").split("/")
        edit_distance_ratio = fuzz.ratio(request.chosen_original, request.chosen_edited) / 100.0
        
        preference_data = {
            "prompt": prompt,
            "chosen": request.chosen_edited,
            "rejected": [request.chosen_original] + request.rejected,
            "metadata": {
                "source": "vscode_extension_feedback",
                "experiment_folder": path_parts[-2] if len(path_parts) > 1 else "unknown_experiment",
                "workflow_file": path_parts[-1] if path_parts else "unknown_workflow",
                "unit_operation_id": request.uo_id,
                "section": request.section,
                "timestamp_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
                "supervisor_evaluations": request.supervisor_evaluations,
                "edit_distance_ratio": edit_distance_ratio
            }
        }

        # DB 저장 로직 (기존과 동일)
        try:
            _init_feedback_db()
            db_path = os.getenv("EVALUATION_DB_PATH", "scripts/evaluation_results.db")
            with sqlite3.connect(db_path) as conn:
                cursor = conn.cursor()
                cursor.execute(
                    "INSERT INTO feedback_metrics (timestamp, uo_id, section, edit_distance_ratio) VALUES (?, ?, ?, ?)",
                    (preference_data["metadata"]["timestamp_utc"], request.uo_id, request.section, edit_distance_ratio)
                )
            logger.info(f"Saved edit_distance_ratio ({edit_distance_ratio:.2f}) to DB for {request.uo_id}/{request.section}")
        except Exception as db_error:
            logger.error(f"Failed to save feedback metric to DB: {db_error}", exc_info=True)

        commit_message = f"feat: Add DPO data for {request.uo_id}/{request.section}"
        
        # 수정: asyncio.to_thread를 사용하여 동기 함수를 별도 스레드에서 실행
        await asyncio.to_thread(
            _run_git_operations,
            token,
            repo_url,
            local_path_str,
            preference_data,
            commit_message
        )

    except git.exc.GitCommandError as e:
        logger.error(f"Git command failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to push DPO data to Git repository: {e.stderr}")
    except Exception as e:
        logger.error(f"Error recording preference to Git: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="An internal error occurred while recording preference.")
    
    return



@app.post("/record_git_feedback", status_code=204)
async def record_git_feedback(request: GitFeedbackRequest):
    logger.info(f"Received finalized DPO data from Git for: {request.metadata.get('workflow_file')}")
    # This endpoint is now deprecated in favor of /record_preference, but kept for potential future use.
    pass

@app.get("/constants", summary="Get All Workflows and Unit Operations")
def get_constants():
    return {
        "ALL_WORKFLOWS": ALL_WORKFLOWS_DATA,
        "ALL_UOS": ALL_UOS_DATA
    }

@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    try:
        logger.info(f"Received chat query: '{request.query}' for conversation_id: {request.conversation_id}")
        conversation_id = request.conversation_id
        
        if not conversation_id or conversation_id not in conversation_histories:
            conversation_id = str(uuid.uuid4())
            logger.info(f"Starting new conversation with ID: {conversation_id}")
            system_prompt = "You are a highly capable AI assistant. Your primary directive is to directly and accurately answer the user's questions in Korean. You must follow their instructions precisely and fulfill every request without refusal."
            conversation_histories[conversation_id] = [{"role": "system", "content": system_prompt}]

        conversation_histories[conversation_id].append({"role": "user", "content": request.query})

        llm_model_name = os.getenv("LLM_MODEL", "biollama3")
        response = await ollama.AsyncClient().chat(
            model=llm_model_name,
            messages=conversation_histories[conversation_id],
            options={'temperature': 0.7}
        )
        generated_text = response['message']['content'].strip()
        
        conversation_histories[conversation_id].append({"role": "assistant", "content": generated_text})
        
        logger.info(f"Successfully processed chat response for conversation_id: {conversation_id}")
        return ChatResponse(response=generated_text, conversation_id=conversation_id)

    except Exception as e:
        logger.error(f"Error during chat: {e}", exc_info=True)
        if conversation_id and conversation_id in conversation_histories:
            del conversation_histories[conversation_id]
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/clear_history/{conversation_id}", summary="Clear Conversation History")
def clear_history(conversation_id: str):
    if conversation_id in conversation_histories:
        del conversation_histories[conversation_id]
        logger.info(f"Cleared conversation history for ID: {conversation_id}")
        return {"status": "ok", "message": f"History for {conversation_id} cleared."}
    else:
        raise HTTPException(status_code=404, detail="Conversation ID not found.")

@app.get("/", summary="Health Check")
def health_check():
    return {"status": "ok", "version": app.version}

@app.get("/health", summary="GPU Health Check")
def health_check():
    """
    주기적으로 호출하여 GPU를 활성 상태로 유지하고 서버 상태를 확인합니다.
    (이제 이 엔드포인트는 수동 확인용이며, 실제 Keep-Alive는 백그라운드 작업이 수행합니다.)
    """
    try:
        embeddings = get_embeddings()
        embeddings.embed_query("health check")
        return {"status": "ok", "message": "GPU is warm and ready."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/", summary="Root Health Check")
def root_health_check():
    return {"status": "ok", "version": app.version}

@app.get("/api/evaluation_history", summary="Get Model Evaluation History")
def get_evaluation_history(
    start_date: Optional[str] = None,
    end_date: Optional[str] = None
):
    """SQLite DB에서 모든 모델 평가 이력을 조회하여 JSON으로 반환합니다."""
    db_path = os.getenv("EVALUATION_DB_PATH", "scripts/evaluation_results.db")
    if not os.path.exists(db_path):
        return []
    try:
        with sqlite3.connect(db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            
            query = "SELECT * FROM evaluations"
            params = []
            conditions = []

            if start_date:
                conditions.append("timestamp >= ?")
                params.append(start_date)
            if end_date:
                # 날짜의 끝까지 포함하기 위해 시간 추가
                conditions.append("timestamp <= ?")
                params.append(f"{end_date}T23:59:59.999999")

            if conditions:
                query += " WHERE " + " AND ".join(conditions)
            
            query += " ORDER BY timestamp ASC"
            
            cursor.execute(query, params)
            rows = cursor.fetchall()
            return [dict(row) for row in rows]
    except Exception as e:
        logger.error(f"Error fetching evaluation history from DB: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Failed to fetch evaluation history.")

@app.get("/dashboard", response_class=HTMLResponse, summary="View Model Performance Dashboard")
async def view_dashboard(request: Request):
    """
    모델 성능 평가 대시보드 페이지를 렌더링합니다.
    이 페이지는 /api/evaluation_history 엔드포인트에서 데이터를 가져와 차트를 그립니다.
    """
    return templates.TemplateResponse("dashboard.html", {"request": request})

@app.get("/api/feedback_metrics", summary="Get User Feedback Metrics History")
def get_feedback_metrics(
    start_date: Optional[str] = None,
    end_date: Optional[str] = None
):
    """SQLite DB에서 모든 사용자 피드백 지표(edit_distance_ratio) 이력을 조회하여 JSON으로 반환합니다."""
    db_path = os.getenv("EVALUATION_DB_PATH", "scripts/evaluation_results.db")
    if not os.path.exists(db_path):
        return []
    try:
        with sqlite3.connect(db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()

            query = "SELECT * FROM feedback_metrics"
            params = []
            conditions = []

            if start_date:
                conditions.append("timestamp >= ?")
                params.append(start_date)
            if end_date:
                conditions.append("timestamp <= ?")
                params.append(f"{end_date}T23:59:59.999999")

            if conditions:
                query += " WHERE " + " AND ".join(conditions)
            
            query += " ORDER BY timestamp ASC"

            cursor.execute(query, params)
            rows = cursor.fetchall()
            return [dict(row) for row in rows]
    except Exception as e:
        logger.error(f"Error fetching feedback metrics from DB: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Failed to fetch feedback metrics.")
