# 2023 기상청 주관 날씨 빅데이터 콘테스트

2023.06.16 ~ 2023.08.08

✓ 프로젝트 소개
------
- 과제 1: 기상에 따른 계절별 지면온도 산출기술 개발 
- 과제 2: 기상에 따른 닻 끌림 예측
  
기상청에서 제시한 다음 2가지의 과제 중, 기상에 따른 계절별 지면 온도 산출기술 개발 진행


✓ 팀 구성 및 역할
------
  - 김수빈 (성균관대학교 통계학과): EDA 및 데이터가 평균으로 보간된 경우 모델링
  - 박시언 (성균관대학교 통계학과): EDA 및 데이터가 평균으로 보간된 경우 모델링
  - 김준서 (성균관대학교 통계학과): EDA 및 데이터가 보간되지 않은 경우 모델링 
  - 조웅빈 (성균관대학교 통계학과): EDA 및 데이터가 보간되지 않은 경우 모델링
  - 서희나 (성균관대학교 통계학과): EDA 및 데이터가 iterative imputer로 보간된 경우 모델링
  - 이지윤 (성균관대학교 통계학과): EDA 및 데이터가 iterative imputer로 보간된 경우 모델링
  
✓ 분석 흐름 
------
<img width="696" alt="스크린샷 2024-01-02 오후 9 38 38" src="https://github.com/jiyunLeeee/2023_Weather_Big_Data_Contest/assets/134356622/5e3564fd-fdf0-49b9-9d75-10dc8ecc1e80">

✓ 분석 결과
------ 
 - Catboost Regressor (XGBoost) + Light GBM stacked hybrid 모형이 가장 좋은 성능을 보여줌
 - 해당 모델의 정확한 지면온도 예측을 통해 농업, 건설업, 에너지 생산 소비, 겨울철 결빙 방지 등의 분야의 발전에 기여할 것이라 예상
 - 현재 기상청 홈페이지에는 지면온도를 제공하고 있지 않아 사람들은 실제 온도, 체감 온도, 지면온도 차이를 비교할 수 없기 때문에  기상청 앱과 웹페이지에 지면온도에 대한 내용 추가를 요망
 - 이를 통해 더위, 추위 등을 대비하는데 도움을 얻을 수 있으며 인적﹒물적 피해를 최소화 할 수 있을 것으로 예상

✓ 사용 Tool
------
<img alt="Python" src ="https://img.shields.io/badge/Python-3776AB.svg?&style=flat-square&logo=Python&logoColor=white"/> <img alt="R" src ="https://img.shields.io/badge/R-276DC3.svg?&style=flat-square&logo=R&logoColor=white"/> 
