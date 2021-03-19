# Form [공식문서](https://docs.djangoproject.com/en/3.1/topics/forms/)

> `Form`은 django의 `유효성 검사 도구` 중 하나이다. 세 가지의 기능을 제공한다.

1. 렌더링을 위한 데이터 구성
2. HTML `form tag` 생성
3. 요청으로 받은 `데이터 처리`

> 요청받은 데이터를 특정 `model`의 레코드로 저장하기 위해서는`.cleaned_data[]` 로 데이터를 추출해서 처리해야한다.

### 1. Form 생성

```python
# 예시 Model
# app/models.py
from django.db import models


class Article(models.Model):
    title = models.CharField(max_length=10)
    content = models.TextField()
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

```

```python
# app/forms.py 
# fields DOC : https://docs.djangoproject.com/en/3.1/ref/forms/fields/
# widgets DOC : https://docs.djangoproject.com/en/3.1/ref/forms/widgets/
from django import forms


class ArticleForm(forms.Form):
    title = forms.CharField(max_length=10)
    content = forms.CharField(widget=forms.Textarea)

```

---

### 2. Form 정의

```python
# app/views.py

from articles.models import Article
from django.shortcuts import render
from .models import Article
from .forms import ArticleForm

# ...

def new(request):
    # Form 인스턴스 변수 생성
    form = ArticleForm
    context = {
        "form": form,
    }
    return render(request, "articles/new.html", context)

```

---

### 3. Form 호출

```django
{% extends 'base.html' %}
{% block content %}
<form action="" method="POST">
  {% csrf_token %}
  {{form.as_p}}
  <button>글작성</button>
</form>
{% endblock content %}
```

---

### 4. 데이터 저장

```python
# app/views.py

# ...

def create(request):
    # 값 받아서 폼 인스턴스 생성
    form = ArticleForm(request.POST)

    # 유효성 검사
    if form.is_valid():
        # 모델 인스턴스 생성 후 데이터 대입 및 저장
        article = Article()
        # .cleaned_data 로 값을 받아서 다시 저장 
        article.title = form.cleaned_data["title"]
        article.content = form.cleaned_data["content"]
        article.save()
        return redirect("articles:index")
    
	# 검사 통과 못했을 때 오류 메세지 전달
    context = {
        "form": form,
    }
    return render(request, "articles:new", context)
```

 



