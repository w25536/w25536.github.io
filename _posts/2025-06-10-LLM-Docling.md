---
title: "[개발일지] Docling으로 PDF 문서 파싱하기: LangChain 연동부터 Qdrant까지"
last_modified_at: 2025-06-10
categories:
  - AI/ML
  - Development
  - Tutorial
tags:
  - Docling
  - LangChain
  - PDF Processing
  - RAG
  - Qdrant
  - Vector Database
  - Document Parsing
  - Python
  - Embedding
excerpt: "Docling 라이브러리를 활용한 PDF 문서 파싱부터 LangChain 연동, Qdrant 벡터DB 구축까지 실전 개발 경험을 공유합니다."
use_math: true
classes: wide
---


![](https://velog.velcdn.com/images/u25536/post/7f5809d8-2e98-462c-a686-48edb1713db2/image.png)


> Docling이란?
Docling은 다양한 문서 형식을 효율적으로 파싱(분석)하고, 구조화된 데이터로 변환해주는 오픈소스 Python 라이브러리입니다. PDF, DOCX, PPTX, 이미지, HTML, AsciiDoc, Markdown 등 여러 문서 포맷을 지원하며, 변환 결과를 Markdown, JSON, HTML 등 다양한 형식으로 내보낼 수 있습니다

개발 관련 내용을 포스팅을 많이 하진 않지만 오늘은 Docling에 대해 잠깐 정리해보겠다. 

LLM 가장 기본적인 문서 로드와 파싱에 대해서부터 설명은 skip하고 바로 본론으로 넘어가자.


지금 하려는 부분은 아래와 동일하다.



<img src="https://velog.velcdn.com/images/u25536/post/ee084a29-6a11-4e7c-add8-c0dbaa5a0790/image.png" width="25%" height="50%">



```python

from langchain_docling import DoclingLoader
from langchain_docling.loader import ExportType

FILE_PATH = "https://arxiv.org/pdf/2408.09869"

# Fix the export_type parameter - JSON is not a valid enum value
# Using a valid ExportType value instead
loader = DoclingLoader(
    file_path=FILE_PATH,
    export_type=ExportType.MARKDOWN,  # Changed from JSON to MARKDOWN
)

docs = loader.load()

for i, doc in enumerate(docs):
    print(f"\nDocument {i}:")
    print("Text:", doc.page_content)
    print("Metadata:", doc.metadata)
    print("-" * 80)

```

![](https://velog.velcdn.com/images/u25536/post/95ea7b26-19ec-4a86-9ecc-79c7b19fa55d/image.png)

코드를 실행시키면 위와 같이 MARKDOWN 형식으로 가져올 수 있고 metadata source가 하나이다 그럴 경우에 qdrant에 page 번호를 담을 수 없기 때문에 추가적으로 해줘야 하는 작업이 있다. 

필자는 JSON 형식으로 추출을 시도하였으나 아래와 같은 오류를 경험했다. 

AttributeError: JSON 오류는 langchain-docling 라이브러리 버전에서 ExportType.JSON이 지원되지 않아 발생합니다. 아래 단계별 해결 방법을 참고하세요.

GPT 답변은 다음과 같았다 'ExportType은 DOC_CHUNKS와 MARKDOWN만 지원'.

DOC_CHUNKS로 바꿔주면 page를 가져올 수 있으니 참고 바란다. 

```python

from docling.document_converter import DocumentConverter
from docling.chunking import HybridChunker

FILE_PATH = "https://arxiv.org/pdf/2408.09869"


# DocumentConverter로 DoclingDocument 얻기
converter = DocumentConverter()
result = converter.convert(FILE_PATH)
dl_doc = result.document  # DoclingDocument 객체

# HybridChunker 사용
chunker = HybridChunker(
    max_tokens=512,
    merge_peers=True
)

chunks = list(chunker.chunk(dl_doc))
```

![](https://velog.velcdn.com/images/u25536/post/bb128b5e-95b6-4866-821d-3551621d40bb/image.png)

필자가 사용한 방법은 위와 같은 방법이다. 초기 방법을 시도하였을 때는 Text Splits 과정을 따로 진행해줘야 했지만 위와 같은 방법을 사용하면 추가적인 Text Splits 과정을 Skip할 수 있다. 

위의 이미지 결과를 보면 page 숫자 필요한 metadata를 가져와 정제 후 Qdrant에 넣어주면 된다. 

다음 할 일은 추출된 다수의 documents들을 qdrant에 때려 넣었을 때 성능 Top-k 방식으로 불러왔을 때 다른 PDF와 Embedding이 겹치지 않는지 필자는 테스트를 진행해보겠다. 



