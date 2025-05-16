---
title: "[LangChain] Retriever 핵심 개념 및 주요 유형 요약"
last_modified_at: 2025-05-16
categories:
  - NLP
  - RAG
  - LangChain
tags:
  - Retriever
  - LangChain
  - RAG
  - VectorStore
  - ParentDocument
  - MultiVector
  - SelfQuery
  - ContextualCompression
  - TimeWeighted
  - MultiQuery
  - Ensemble
excerpt: "[LangChain] Retriever핵심 개념 및 주요 유형 요약"
use_math: true
classes: wide
---

필자는  https://databoom.tistory.com/entry/Langchain-Retriever-%EC%82%AC%EC%9A%A9%ED%95%98%EA%B8%B0 아래 내용을 가져 왔다. 앞으로 이 내용을 기준으로 제가 사용하는 방법도 순차적으로 추가해보겠다. 

## Retriever 종류 요약

| Retriever 종류                        | 분류  | 내용                                                                                                                                                                                                                                           |
| ----------------------------------- | --- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Vectorstore Retriever               | 설명  | 각 텍스트 조각마다 임베딩을 생성하여 검색을 수행<br>주로 유사도 검색을 위해 벡터 공간에 문서를 매핑하여 활용                                                                                                                                                                              |
|                                     | 장점  | 가장 기본적이고 쉽게 시작                                                                                                                                                                                                                               |
| ParentDocument Retriever            | 설명  | 대규모 문서를 상위 문서 단위로 묶고, 맥락을 보존한 채 검색을 단순화하려는 시나리오<br>임베딩 공간에서 가장 유사한 청크를 찾은 후 해당 청크가 속한 전체 문서를 반환                                                                                                                                              |
|                                     | 장점  | - 긴 문서를 관리해야 할 때 유용<br>- 전체 문서를 검색할 때 유용.<br>- 검색 시, 상위 문서만 확보하면 되기에 적용 로직이 단순<br>- 문서의 '맥락'을 그대로 유지해야 할 때 유용<br>- 낮은 노이즈(자잘하게 쪼개진 문서들에서 발생할 수 있는 중복 검색 결과나 불필요한 노이즈를 줄임)                                                                    |
|                                     | 한계  | - 문서의 길이가 너무 짧은 데이터가 많을때, 문서의 개수가 적을 때에는 필요없음<br>- 아주 세밀한 문장 단위 검색이 필요한 경우는 부적합함<br>- 사용자가 특정 섹션 정보만 필요하고, 다른 섹션은 크게 의미가 없는 질의를 한다면, 굳이 '전체 문서' 또는 그 상당 부분을 반환할 필요가 없음 (질의 자체가 특정 섹션에 강하게 종속되어 있을 때)<br>- 세밀한 Ranking, Scoring이 중요한 경우       |
| Multi Vector Retriever              | 설명  | 같은 문서에 여러 개의 벡터를 생성하는 방식                                                                                                                                                                                                                     |
|                                     | 장점  | - 한 문서가 여러 주제를 다루거나, 다양한 내용이 뒤섞여 있는 경우 좋음<br>- 세밀한 문단/문장 단위 검색이 필요한 경우<br>- 정밀도가 중요한 검색 또는 Q&A 시스템<br>- 특화된 임베딩 조합 가능(기술 용어가 잦은 구간, 숫자·통계 데이터가 주된 구간, 문어체·구어체 구간 등)<br>- 길이가 긴 문서 처리에 유리(길이가 매우 긴 문서를 단일 벡터로 만들면 임베딩 하나가 거대 -> 유사도 검색시 힘들어짐) |
|                                     | 한계  | - 리소스(메모리, 인덱스 크기)와 운영 비용이 많이듦<br>- 문서 자체가 너무 짧거나 단일 토픽으로만 구성된 경우 굳이 필요없음<br>- 구현하기가 복잡함<br>- 검색 속도가 느려짐                                                                                                                                     |
| Self Query Retriever                | 설명  | LLM을 사용해 사용자의 질문을 자체를 분석·변환<br>주로 질문이 문서의 메타데이터와 관련된 경우, LLM이 질문을 검색할 문자열과 메타데이터 필터로 변환. 해당 조건에 맞춰 검색을 수행                                                                                                                                    |
|                                     | 장점  | - 문서 메타데이터나 구조화된 필드가 있는 경우, 조건으로 필터링 가능<br>- 질의 의도를 깊이 이해, 정교한 검색 정확도<br>- 확장성 (문서 메타데이터(작성자, 날짜, 토픽, 지역, 버전 등)가 늘어나더라도, 모델이 질의 문장에서 이를 찾아낼 수만 있다면, 새로운 필터 타입도 자동으로 대응 가능)                                                                   |
|                                     | 한계  | - 구조화된 필드나 메타데이터가 거의 없는 경우, 사용 불가능<br>- LLM(대형 언어모델) 적용 비용/복잡도가 부담스러운 경우 어려움<br>- 질의 구조가 거의 단순하거나, 필터 없이 전체에서 찾는 단순한 검색이 주된 경우 필요없음                                                                                                          |
| Contextual Compression Retriever    | 설명  | 문서 압축 기법<br>검색된 문서에 불필요한 정보가 너무 많을 때 유용<br>검색 후, 후처리 과정으로 LLM 또는 임베딩을 사용해 가장 중요한 정보만 추출                                                                                                                                                      |
|                                     | 장점  | - 문서가 길고, 사용자가 원하는 정보는 그 일부일 때 유용<br>- LLM 처리 비용(토큰 수)이 중요한 경우 절약 가능<br>- 노이즈가 많은 데이터 필터링<br>- 정확하고 간결한 요약이 필요한 상황                                                                                                                           |
|                                     | 한계  | - 문서가 원체 짧거나, 압축이 필요 없는 상황은 필요없음'<br>- 질문 맥락이 광범위해, 문서의 여러 부분이 두루두루 관련될 때, 요약해도 압축이 안됨<br>- 잘못된 압축으로 정답의 근거가 사라지면, 최종 답변이 틀릴 수 있다<br>- 추출·요약 과정 자체가 비싸거나 복잡해질 수 있음                                                                           |
| Time-Weighted Vectorstore Retriever | 설명  | 최신성 기준을 결합하여 검색<br>최신 문서 검색이 중요한 경우에 유용                                                                                                                                                                                                      |
|                                     | 장점  | - 뉴스 기사, 소셜 미디어 포스트, 기술 문서(버전이 자주 업데이트됨) 등은 최근 정보가 중요한 경우 유용, 자주 업데이트되거나 변동이 큰 문서를 다뤄야 할 때<br>- '가장 최신' 또는 '가장 최근의 이벤트'가 핵심인 질의가 자주 발생할 때                                                                                                    |
|                                     | 한계  | - 문서 업데이트 빈도가 낮고, 대부분 오래된 문서로 구성된 데이터<br>- 정확한 시간 정보(타임스탬프)가 없는 데이터셋은 못씀                                                                                                                                                                     |
| Multi-Query Retriever               | 설명  | LLM을 사용해 원래 질문에서 여러 개의 쿼리를 생성<br>각 쿼리에 대해 문서를 검색한 후 이를 조합                                                                                                                                                                                    |
|                                     | 장점  | - 질의가 포괄적이거나, 여러 표현이 가능할 때<br>- 다양한 표현(동의어, 유의어) 커버<br>- 검색 누락(정보 손실) 감소                                                                                                                                                                     |
|                                     | 한계  | - 질의가 매우 구체적이고 단순할 때, 필요없음<br>- 리소스(시간, 비용) 제한이 엄격한 경우<br>- 검색 노이즈가 크게 증가할 위험이 있음                                                                                                                                                            |
| Ensemble Retriever                  | 설명  | 여러 가지 검색 방식을 결합<br>다양한 retriever를 사용해 각기 다른 방식으로 문서를 검색, 이를 합쳐서 최종적으로 문서를 반환                                                                                                                                                                 |
|                                     | 장점  | 다양한 방법을 취합해 성능 고도화                                                                                                                                                                                                                           |
|                                     | 단점  | 리소스 과소비, 느려질 수 있음                                                                                                                                                                                                                            |
| Long-Context Reorder Retriever      | 설명  | 긴 문맥을 처리하는 모델에서 중간 정보를 무시하는 경향이 있을 때 사용<br>검색된 문서를 재배열하여 가장 유사한 정보를 처음과 끝에 배치하여, 긴 문맥에서 유용한 정보를 놓치지 않도록 한다.                                                                                                                                  |
|                                     | 장점  | 간단한 구현                                                                                                                                                                                                                                       |
|                                     | 단점  | - 모델별 편향적임 (모든 모델이 맨 끝의 내용을 중요하게 반영하는지 확인 필요)<br>- 맥락 훼손 가능성<br>- 청크 선별 정확도가 보장 되어야 함                                                                                                                                                        |




# 3. Retriever 만들기

## 3.1 Vectorstore Retriever

**벡터 기반 Retriever**: 베이스가 되는 Retriever

`vectorstore.as_retriever()` 메서드로 생성

**예시 코드 (FAISS 사용)**

```python
from langchain.vectorstores import FAISS

vectorstore = FAISS.from_documents(docs, embeddings)
retriever = vectorstore.as_retriever()
```


**전체 코드**  
실행 순서:

1. 문서를 청크로 분할 (`CharacterTextSplitter`)
    
2. 임베딩 생성 (`OpenAIEmbeddings`)
    
3. 벡터스토어 구축 (`FAISS`)
    
4. Retriever 정의 (`vectorstore.as_retriever()`)
    

```python
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.text_splitter import CharacterTextSplitter
from langchain.document_loaders import TextLoader
from langchain.chains import RetrievalQA
from langchain.llms import OpenAI

# 1. 문서 로드
loader = TextLoader('your_document.txt')
documents = loader.load()

# 2. 텍스트 분할
text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
docs = text_splitter.split_documents(documents)

# 3. 임베딩 및 벡터스토어 생성
embeddings = OpenAIEmbeddings()
vectorstore = FAISS.from_documents(docs, embeddings)

# 4. Retriever 설정
retriever = vectorstore.as_retriever()

# 5. QA 체인 생성
qa = RetrievalQA.from_chain_type(
    llm=OpenAI(),
    chain_type="stuff",
    retriever=retriever
)

# 6. 질문에 답변
query = "여기에 질문을 입력하세요."
answer = qa.run(query)
print(answer)
```

---

## 3.2 Self Query Retriever

**설명**: 질문에 대한 메타데이터를 함께 생성해주는 Retriever

```python
from langchain.retrievers import SelfQueryRetriever, AttributeInfo
from langchain_chroma import Chroma
from langchain.embeddings import OpenAIEmbeddings
from langchain.llms import OpenAI
from langchain.schema import Document

# 문서 생성
documents = [
    Document(page_content="이 문서는 과학에 관한 내용입니다.", metadata={"title": "과학", "year": 2025}),
    Document(page_content="이 문서는 예술에 관한 내용입니다.", metadata={"title": "예술", "year": 2024}),
]

# 메타데이터 정의
metadata_field_info = [
    AttributeInfo(
        name="title",
        description="문서의 카테고리. ['과학', '예술'] 중 하나",
        type="string",
    ),
    AttributeInfo(
        name="year",
        description="문서가 작성된 연도",
        type="integer",
    ),
]

# 벡터스토어 생성
embeddings = OpenAIEmbeddings()
vectorstore = Chroma.from_documents(documents, embeddings)

# Self Query Retriever 설정
llm = OpenAI()
retriever = SelfQueryRetriever.from_llm(
    llm=llm,
    vectorstore=vectorstore,
    metadata_field_info=metadata_field_info
)
```

---

## 3.3 Contextual Compression Retriever

**설명**: 문서 길이가 너무 길 때, 핵심 부분만 압축하여 검색

```python
from langchain.retrievers import ContextualCompressionRetriever
```

### 3.3.1 LLMChainExtractor 사용

```python
from langchain.retrievers.document_compressors import LLMChainExtractor
from langchain_openai import OpenAI

llm = OpenAI(temperature=0)
compressor = LLMChainExtractor.from_llm(llm)
compression_retriever = ContextualCompressionRetriever(
    base_compressor=compressor,
    base_retriever=retriever
)
```

### 3.3.2 LLMChainFilter 사용

```python
from langchain.retrievers.document_compressors import LLMChainFilter

_filter = LLMChainFilter.from_llm(llm)
compression_retriever = ContextualCompressionRetriever(
    base_compressor=_filter,
    base_retriever=retriever
)
```

### 3.3.3 LLMListwiseRerank 사용

```python
from langchain.retrievers.document_compressors import LLMListwiseRerank
from langchain_openai import ChatOpenAI

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
_filter = LLMListwiseRerank.from_llm(llm, top_n=1)
compression_retriever = ContextualCompressionRetriever(
    base_compressor=_filter,
    base_retriever=retriever
)
```

### 3.3.4 EmbeddingsFilter 사용

```python
from langchain.retrievers.document_compressors import EmbeddingsFilter
from langchain_openai import OpenAIEmbeddings

embeddings = OpenAIEmbeddings()
embeddings_filter = EmbeddingsFilter(
    embeddings=embeddings,
    similarity_threshold=0.76
)
compression_retriever = ContextualCompressionRetriever(
    base_compressor=embeddings_filter,
    base_retriever=retriever
)
```

---

## 3.4 Time-Weighted Vectorstore Retriever

**설명**: 시간에 따른 가중치를 적용하여 최신 문서를 우선 검색

```python
from langchain.retrievers import TimeWeightedVectorStoreRetriever
from langchain.vectorstores import FAISS
from langchain.embeddings import OpenAIEmbeddings
from langchain.schema import Document
import datetime

# 문서에 타임스탬프 추가
documents = [
    Document(page_content="2021년의 정보입니다.", metadata={"timestamp": datetime.datetime(2021, 1, 1)}),
    Document(page_content="현재의 정보입니다.", metadata={"timestamp": datetime.datetime.now()}),
]

embeddings = OpenAIEmbeddings()
vectorstore = FAISS.from_documents(documents, embeddings)

retriever = TimeWeightedVectorStoreRetriever(
    vectorstore=vectorstore,
    decay_rate=0.1,
    time_key="timestamp"
)

# 검색 예시
query = "최신 정보를 알려주세요."
docs = retriever.get_relevant_documents(query)
for doc in docs:
    print(doc.page_content)
```

---

## 3.5 Multi-Query Retriever

**설명**: 여러 쿼리를 생성하여 합집합으로 검색

```python
from langchain.retrievers.multi_query import MultiQueryRetriever
from langchain_openai import ChatOpenAI
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma

# VectorDB 생성
embedding = OpenAIEmbeddings()
vectordb = Chroma.from_documents(documents=splits, embedding=embedding)

question = "What are the approaches to Task Decomposition?"
llm = ChatOpenAI(temperature=0)
retriever_from_llm = MultiQueryRetriever.from_llm(
    retriever=vectordb.as_retriever(),
    llm=llm
)
```

---

## 3.6 Ensemble Retriever

**설명**: 여러 Retriever 결과를 가중합하여 결합

```python
from langchain.retrievers import EnsembleRetriever

retriever1 = vectorstore1.as_retriever()
retriever2 = vectorstore2.as_retriever()

ensemble_retriever = EnsembleRetriever(
    retrievers=[retriever1, retriever2],
    weights=[0.5, 0.5]
)

# 검색 예시
query = "통합된 검색 결과를 보여주세요."
docs = ensemble_retriever.get_relevant_documents(query)
for doc in docs:
    print(doc.page_content)
```

---

## 3.7 Long-Context Reorder Retriever

**설명**: 검색된 문서들의 순서를 재정렬

```python
from langchain_community.document_transformers import LongContextReorder

reordering = LongContextReorder()
docs = retriever.invoke(query)
reordered_docs = reordering.transform_documents(docs)
```

---

# 4. 하이브리드 검색 (키워드 검색)

키워드 기반 검색 기능 (벡터DB 지원 여부 확인 필요)  
Astra DB 벡터스토어의 `body_search`로 필터링 가능

```python
from langchain.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain.schema import StrOutputParser, RunnablePassthrough, ConfigurableField

template = """Answer the question based only on the following context:
{context}
Question: {question}
"""
prompt = ChatPromptTemplate.from_template(template)
model = ChatOpenAI()
retriever = vectorstore.as_retriever()

# Configurable retriever 설정
configurable_retriever = retriever.configurable_fields(
    search_kwargs=ConfigurableField(
        id="search_kwargs",
        name="Search Kwargs",
        description="키워드 검색 인자",
    )
)

chain = (
    {"context": configurable_retriever, "question": RunnablePassthrough()}
    | prompt
    | model
    | StrOutputParser()
)

# 실행 시 키워드 등록
chain.invoke(
    "What city did I visit last?",
    config={"configurable": {"search_kwargs": {"body_search": "new"}}},
)
```

---

# 5. 실제 프로젝트에서 필요한 사항 정리

- **서비스 단계**: Vectorstore Retriever로 시작하거나, 상황에 맞는 Retriever 선택
    
- **Retriever 성능 평가**: 문서를 제대로 검색하는지 검증
    
- **운영 단계**: 대규모 채팅 데이터·벡터스토어 분석 후 최적의 Retriever로 조정
    
- **채팅 데이터 분석**: 주제별 출현 빈도 파악
    
- **벡터스토어 분석**: 수집 데이터 변화 여부 확인 (초기 가정 대비)