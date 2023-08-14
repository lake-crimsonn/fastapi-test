# 230812

## 수업

### 미니포지

- 오픈소스라서 미니콘다와 다르게 라이센스 문제를 피할 수 있다.
- _pip와 conda 차이_
  - pip는 파이선의 지원을 받는 패키지. conda는 anaconda에 지원을 받는 패키지
  - pip가 구글플레이스토어라고 하면, conda는 갤럭시 삼성 스토어

### ai 이슈

- ITworld CIO Market Pulse
- ai를 점점 도입하고 있는 회사들이 늘어나고 있다. 대기업일수록 이미 도입하고 있는 경우가 많다.
- 2021년에 tansformer와 bert가 나온 이후 nlp가 많이 뜨고 있다.
- 이미지와 같은 시각적인 ai 분야는 저작권이나 초상권때문에 서비스로 상용하기가 어렵다.
- 항상 포트폴리오는 트렌디한 기술을 사용해서 만들자.
- 솔트룩스라는 nlp 전문 회사가 있다.
- ai 솔루션은 앞으로 오픈소스가 많이 늘어나고, 자체 개발은 줄어들 예정이다.
- rpa는 업무자동화프로그램이다. 대표적으로 uipath가 있다.
- ai 엔지니어나 mlops는 전통적으로 데이터부족와 데이터품질이 좋지 않기 때문에 점점 뜨고 있는 직업이다. 라마나 chatgpt를 이용한 개발을 해야한다.
- 6천만원 gpu h100을 사려면 34주를 기다려야 한다. 그래서 자체개발은 점점 더 어려워진다.

### _l4 l7 로드밸런서 차이_

- 알아보기

### ai serving 이론

- stand alone app: hardware - os - app - user
- api는 데이터를 주고받는 기계가 서로 달라도 작동할 수 있도록 도와준다.
- ai서버는 메인서버와 분리돼서 운영이 되는 구조를 마이크로서비스라고 한다.
- ai 모델 서버 = api서버
- _모델서빙 디자인패턴_
  - client - load balancer - several ai servers
  - web singleton pattern: 가장 기본적인 서버의 형태. 이 이후의 패턴은 싱글톤 패턴이 감당하기 어려운 요청이 있을 때 사용한다.
  - 파이썬과 cpp은 바인딩하여 하드웨어의 리소스를 직접 사용한다. 자바는 jvm이 하드웨어를 다루기에는 유틸리티가 부족하여 어렵다고 한다.
  - ai모델서버를 바로 이용하면 안되는 이유: 허가되지 않는 사용자에 대한 요청, ddos에 대해 약함.
- Synchronous pattern: 디폴트
- Preprocess-prediction pattern: gpu서버를 놀지 않게 하는 컨셉
- microservice vertical pattern: 모델 추론 서버가 여러개(랜드마크, 얼굴인식 등) - 프록시, 하나하나 의존적이어서 순서대로 해야되서 문제
- 파이썬 언어만 쓴다면? 도커 - 구니콘(파이썬은 gli때문에 멀티 쓰레드 불가해서 동시처리용 멀티프로세스는 가능) - 유비콘(구니콘 띄우는 유비콘) - fastapi - deeplearning framework - model
- 텐서플로우 서빙은 cpp로 껍데기(구니콘 유니콘 fastapi)를 대처할 수 있고 빠르다고 한다.
- 파이토치 서브는 cpp 내부, 껍데기는 자바. 잘 사용하지 않는다.
- nvidia가 만든 Triton은 파이토치나 텐서플로우를 모두 지원한다. 그리고 가장 빠르다.
- bentoml은 파이썬으로 만들어서 속도가 느리다. 기능적인 면에서는 좋다. 딥러닝 프레임워크 모두 지원.
- 기능
  - 멀티플 딥러닝 프레임워크 지원
  - concurrent model execution
  - dynamic batching
  - 속도 관련 메트릭 제공
- concurrent model execution
  - model를 gpu 병렬화해서 실행
  - gpu 코어는 cpu 코어와 다르다. 병렬적으로 처리를 하는 것처럼 보인다. 멀티프로세싱이 가능 하지 않지만, 스트림을 이용한다.
  - 하나의 컴퓨터에 여러개의 모델의 띄워두기 때문에 gpu 메모리 병목현상이 일어남
  - 구니콘은 이슈가 있다 - 도커 컴포즈로 리플리카를 nginx로 로드밸런싱
- 다이나믹 배칭
  - 배치단위로 추론을 함. 요청이 어느정도 쌓일 때까지 기다렸다가 한번에 추론함. 문제는 모은 요청을 다시 분리해서 보내주는 작업이 어려움.
  - 단일 gpu에서는 성능이 좋지만, 멀티 gpu에서는 다이나믹 배칭을 하기 어려움.
- 트라이톤은 무조건 전처리후처리, 추론서버를 나눠야 한다.
- 모델 서빙의 핵심은 다이나믹 배칭과 컨커런트 모델 익스큐션이다.
- 파이트라이톤의 껍데기 cpp 모델, 프레임워크는 파이썬.
- 트라이톤은 껍데기는 cpp, 모델 프레임워크도 cpp로 제일 빠르게 돌릴 수 있으나, 사용자가 적어서 파이트라이톤을 만듦.
- 이제 파이썬에서 gli 없앤다고 한다.

### ai serving 실습

- 코드에디터와 터미널의 가상환경은 서로 다를 수 있다. 오른쪽 하단을 참고
- conda env list: 가상환경 리스트
- conda list: 패키지 리스트
- `conda install pytorch torchvision torchaudio pytorch-cuda=11.7 -c pytorch -c nvidia`: cuda pytorch 설치
- transformer와 pytorch는 호환성이 좋다고 한다.
- openai platform 가입하기
- **QWEN이 라마보다 한국어 이해도가 좋다.**
- `pip install transformers`
- 가상환경 복제
  - 재사용하는 기본 패키지를 클론할 수 있다.
  - `conda create -n test1 --clone test`

### fastapi

- fastapi는 공식페이지에 사용하는 방법이 잘 정리되어 있어서 빠르게 배울 수 있다.
- flask 대비 편리한 기능이 많다. 이미지 페이로드 처리할 때와 같은 경우.
- gli을 사용하지 않아서 멀티프로세스가 안되는데, 싱글스레드 비동기 방식으로 사용이 가능하다. node.js처럼.
- 좋은 프레임워크는 공식 문서가 잘 되어있다. fastapi 역시 공식문서가 잘 정리되어있다.
- `pip install fastapi`
- `pip install "uvicorn[standard]"`
- 구니콘은 윈도우에서 설치가 되지 않는다. 도커에 설치할 예정.
- 기본 코드

  ```python
  from typing import Union

  from fastapi import FastAPI

  app = FastAPI()
  @app.get("/")
  def read_root():
      return {"Hello": "World"}

  @app.get("/items/{item_id}")
  def read_item(item_id: int, q: Union[str, None] = None):
      return {"item_id": item_id, "q": q}
  ```

- `uvicorn main:app --reload` 서버시작
- 리로드 옵션은 매번 모델을 불러오기 때문에 오버헤드가 일어날 수 있다.
- post는 기본적으로 form을 통해 html의 body 데이터를 전송한다.
- `pip install python-multipart` 폼데이터 이용하기 위해 설치.
- swagger가 정말 유용하다. `localhost:8000/docs`
  - 로그인을 테스트할 수 있다.
    ```python
    @app.post("/login/")  # python 3.6+ non_annotated
    async def login(username: str = Form(), password: str = Form()):
        print(password)
        return {"username": username}
    ```
- 모델을 먼저 선언하고 서버코드를 쓰기
- swagger(docs)는 postman과 비슷해 보인다.
- fastapi 파일은 bytes와 fileupload 타입이 있다.
  - bytes는 파일 그 자체와 파일사이즈만 보낸다.
  - fileupload는 파일 메타데이터
  - 비동기로 파일을 받아오기 때문에 병렬처리가 가능하다.
  - 파일을 받으면서 다른 파일의 메타데이터를 받을 수 있다.

### 도커

- 로컬에 있는 쿠다와 같은 엔비디아를 도커 이미지로 넣기는 힘들다.
- 엔디비아에서 제공해주는 이미지를 사용하자.
- nvidia catalog에서 쿠다 이미지를 가져오자.
- 기본적으로 pytorch를 해준다.
- 도커는 제품이름이고, 컨테이너라는 기술이다. containerd
- 도커엔진은 원래 리눅스에서만 동작한다. 윈도우에서 도커를 이용하는 경우 데스크탑을 이용해야 한다.
- 리눅스는 아나콘다를 잘 사용하지 않는다. 이유는?
- 윈도우 서브시스템 리눅스가 윈도우 위에 리눅스를 돌릴 수 있게 해준다. 예전에는 hyerV를 이용했다.
- 사내에서 회사 GPU 서버를 함께 이용한다. 서버를 깨끗하게 유지해야 하니까 커스텀화 된 프로그램을 설치 하면 안된다. 도커 이미지로 모델을 가져와서 공용 GPU를 사용한다.
- 도커 컨테이너 개념 https://www.samsungsds.com/kr/insights/220222_kubernetes1.html
- https://docs.docker.com/get-started/02_our_app/
- 도커 데스크탑 설치
- `git clone https://github.com/docker/getting-started-app.git`
- Dockerfile 파일 생성, 확장자 없음
- vscode Docker extension 설치
- Dockerfile

  ```bash
  # syntax=docker/dockerfile:1

  FROM node:18-alpine
  WORKDIR /app
  COPY . .
  RUN yarn install --production
  CMD ["node", "src/index.js"]
  EXPOSE 3000
  ```

  - WORKDIR 시작 디렉토리
  - node:18: 사용하는 프로그
  - copy . . : 왼쪽 내 디렉토리 경로의 파일을 오른쪽 도커 경로 WORKDIR에 복사를 해줘
  - yarn install --production: package.json을 설치해줘
  - CMD: 리눅스에서 node 실행 명령
  - 콘다 클론과 비슷한 상황
  - 각 코드들을 레이어라고 함

- `docker build -t getting-started .`: Dockerfile 내용에 따라 새로운 이미지를 생성, -t는 태그, 마지막 .은 도커파일이 있는 디렉토리의 위치
- docker images: 이미지 확인
- docker ps: 실행 중인 컨테이너 확인
- docker rm -r [docker-id]: 도커 삭제

#### 모델 추론 서버 fastapi에 적용

- ngc pytorch 구글 검색해서 페이지 들어간 다음에 get container 찾기
- Dockerfile를 fastapi 디렉토리에 생성
- 파이토치와 쿠다가 설치 되어 있는 이미지

  ```
  FROM nvcr.io/nvidia/pytorch:23.07-py3
  WORKDIR /app
  COPY . .
  ```

- 이 이미지는 8기가나 된다.
- 빌드를 해도 되고, 이미지를 도커 풀로 땡겨올 수 있다.
- `docker pull nvcr.io/nvidia/pytorch:23.07-py3`

#### 간단한 fastapi 올리기

- https://fastapi.tiangolo.com/ko/deployment/docker/

- 기본 코드
  ```
  FROM python:3.9
  WORKDIR /code
  COPY ./requirements.txt /code/requirements.txt
  RUN pip install --no-cache-dir --upgrade -r /code/requirements.txt
  COPY ./app /code/app
  CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "3000"]
  EXPOSE 3000
  ```

#### 컨테이너에서 허깅페이스 모델 다루기

- 컨테이너는 껐다가 키고를 반복함. 모델을 매번 다운로드 할 수 없음.
- 컨테이너가 내부 서버에 있다면 외부의 허깅페이스 모델을 다운로드 받을 수 없음.
- 허깅페이스에서 다운로드 받은 경로의 모델 스냅샷을 models 디렉토리에 포함해줘야 한다!!
  - 스냅샷은 json 파일 두개와 바이너리 파일 하나
- `COPY ./models /code/models` dockerfile
- .gitignore에 `models/` 입력
- 모델의 이름이 아니라 모델의 경로를 넣어주기
- 도커파일은 자식 폴더에만 접근이 가능하다.

---
