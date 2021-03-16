# Model Form [공식문서](https://docs.djangoproject.com/en/3.1/topics/forms/modelforms/)



### 1. Model Form 생성

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
from django import forms
from .models import Article


class ArticleForm(forms.ModelForm):
    # 세부정보
    class Meta:
        # 참고할 Model
        model = Article
        # 입력받을 field 설정
        fields = ("title", "content")
        
        # 전체 field 설정
        # fields = "__all__"

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

### 4. 데이터 저장 `Create`

```python
# app/views.py

# ...

def create(request):
    # 값 받아서 폼 인스턴스 생성
    form = ArticleForm(request.POST)

    # 유효성 검사
    if form.is_valid():
        # Model Form 저장 및 인스턴스 변수 생성(detail redirect 하기 위해)
        article = form.save()
        return redirect("articles:index")

    context = {
        "form": form,
    }
    return render(request, "articles:new", context)
```

---

### 5. 데이터 수정 `Update`

```python
# app/views.py

# ...

def edit(request, pk):
    article = Article.objects.get(pk=pk)
    if request.mehtod == "POST":
        form = ArticleForm(request.POST, instance=article)
        if form.is_valid():
            article = form.save()
            return redirect("articles:index")

    else:
        form = ArticleForm(instance=article)

    context = {
        "form": form,
        "article": article,
    }
    return render(request, "articles/edit.html", context)
```

```django
{% extends 'base.html' %}
{% block content %}
<form action="{% url 'articles:edit' article.id  %}" method="POST">
  {% csrf_token %}
  {{form.as_p}}
  <button>수정완료</button>
</form>
{% endblock content %}
```

---

### 속성`attrs` 추가

```python
# app/forms.py  
from django import forms
from .models import Article


class ArticleForm(forms.ModelForm):
    title = forms.CharField(
        widget=forms.TextInput(
            attrs={
				'class': 'my-title abc',
                'placeholder': '10자 이내로 작성해주세요.'
            }
        )
    )
    # 세부정보
    class Meta:
        # 참고할 Model
        model = Article
        # 입력받을 field 설정
        fields = ("title", "content")
        
        # 전체 field 설정
        # fields = "__all__"

```

