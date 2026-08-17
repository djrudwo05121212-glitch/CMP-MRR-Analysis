# CMP MRR 예측 모델 및 Sensor 조건 분석

2016 PHM Data Challenge의 CMP(Chemical Mechanical Planarization) 데이터를 이용해 MRR(Material Removal Rate, 연마 제거율)을 분석한 프로젝트입니다.

## 분석 목표

1. Stage–Chamber별 과거 MRR 중앙값을 분석용 Target으로 설정
2. 동일 공정 조건 안에서 MRR 예측 회귀모델 비교
3. 목표 MRR 부근에서 관측된 주요 Sensor 조건 후보 도출

## 분석 대상

| 공정 조건 | 표본 수 | 적용 여부 |
|---|---:|---|
| Stage A–Chamber 1 | 363 | 본 분석 포함 |
| Stage A–Chamber 4 | 795 | 본 분석 포함 |
| Stage B–Chamber 4 | 810 | 본 분석 포함 |
| 기타 Stage–Chamber | 9 | 표본 부족으로 제외 |

## 핵심 결과

| 공정 조건 | 분석용 Target MRR | 선정 모델 | 시험 RMSE | 시험 MAE | 시험 R² |
|---|---:|---|---:|---:|---:|
| Stage A–Chamber 1 | 151.07 | Random Forest | 3.18 | 2.54 | 0.301 |
| Stage A–Chamber 4 | 74.12 | Random Forest | 2.83 | 2.13 | 0.805 |
| Stage B–Chamber 4 | 81.51 | XGBoost | 3.54 | 2.58 | 0.827 |

Stage A–Chamber 1은 시험 R²가 낮아 추가 Sensor·Recipe 정보 없이 실제 운전 판단에 적용하기 어렵습니다.

## 분석 방법

- 동일 Wafer와 Stage의 시계열 Sensor 값을 평균·표준편차·범위로 요약
- Ridge, Random Forest, Extra Trees, HistGradientBoosting, XGBoost, CatBoost 비교
- 3-Fold 교차검증 평균 RMSE를 기준으로 파라미터 후보 선택
- 독립 시험 데이터에서 RMSE·MAE·R² 평가
- 순열중요성과 Tukey Box Plot으로 주요 Sensor 분포 확인

## 폴더 구성

```text
presentation/  최종 PowerPoint 보고서
results/       주요 분석 결과 CSV
src/           분석 및 보고서 생성 코드
```

## 해석 시 주의사항

- Target MRR은 제품 Spec이 아니라 과거 데이터 중앙값에 기반한 분석 기준입니다.
- Sensor 후보 범위는 최적 Setpoint가 아닙니다.
- 공개 데이터에는 Sensor 단위와 상세 Recipe가 제공되지 않습니다.
- 순열중요도와 Box Plot은 관측상 연관성을 나타내며 인과관계를 입증하지 않습니다.
- 실제 적용 전 DOE(실험계획법)와 공정 Spec 검증이 필요합니다.

## 데이터 출처

- PHM Society, 2016 PHM Data Challenge: CMP Data Set
- 원본 데이터는 저장소에 포함하지 않습니다.

