# [django_authentication](https://docs.djangoproject.com/en/3.1/topics/auth/)

## 1. App 생성 & 설정

> 일반적으로 django의 auth 관련 app 이름은 accounts 로 한다.

```bash
python manage.py startapp accounts
```

``` python
# settings.py

INSTALLED_APPS = [
    'accounts',
# ...
]
```

```python
# 설정폴더/urls.py

from django.urls import path,include

urlpatterns = [
    path('accounts/', include('accounts.urls')),
]
```

```python
# accounts/urls.py  

from django.urls import path
from . import views

app_name = 'accounts'
urlpatterns = [
    
]
```

## 2. 마이그레이트

```bash
python manage.py migrate
```

## 3. 유저 목록 조회

```python
# accounts/urls.py

from django.urls import path
from . import views

app_name = "accounts"
urlpatterns = [
    path("", views.index, name="index"),
]
```

```python
# accounts/views.py

from django.shortcuts import redirect, render
from django.contrib.auth import get_user_model


def index(request):
    users = get_user_model().objects.all()
    context = {
        "users": users,
    }
    return render(request, "accounts/index.html", context)
```

```django
{# accounts/templates/accounts/index.html #}
{% extends 'base.html' %}

{% block content %}

{% for user in users %}
<p>{{ user.username }}</p>
{% endfor %}

{% endblock content %}
```

## 4. 회원가입
```python
# accounts/urls.py

from django.urls import path
from . import views

app_name = "accounts"
urlpatterns = [
    # ...
    path("signup/", views.signup, name="signup"),
]
```

```python
# accounts/views.py

from django.shortcuts import redirect, render
from django.contrib.auth import get_user_model
from django.contrib.auth.forms import UserCreationForm
from django.contrib.auth import login as auth_login


def signup(request):
    if request.method == "POST":
        form = UserCreationForm(request.POST)
        if form.is_valid():
            user = form.save()
            auth_login(request, user)
            return redirect("accounts:index")

    else:
        form = UserCreationForm

    context = {
        "form": form,
    }
    return render(request, "accounts/signup.html", context)
```

```django
{# accounts/templates/accounts/signup.html #}
{% extends 'base.html' %}

{% block content %}
<form action="" method="POST">
  {% csrf_token %}
  {{ form.as_p }}
  <button>회원가입</button>
</form>
{% endblock content %}
```
## 5. 로그인
```python
# accounts/urls.py

from django.urls import path
from . import views

app_name = "accounts"
urlpatterns = [
    # ...
    path("login/", views.login, name="login"),
]
```

```python
# accounts/views.py

from django.shortcuts import redirect, render
from django.contrib.auth import get_user_model
from django.contrib.auth import login as auth_login
from django.contrib.auth.forms import UserCreationForm
from django.contrib.auth.forms import AuthenticationForm


def login(request):
    if request.method == "POST":
        form = AuthenticationForm(request, request.POST)
        if form.is_valid():
            auth_login(request, form.get_user())
            return redirect("accounts:index")

    else:
        form = AuthenticationForm

    context = {
        "form": form,
    }
    return render(request, "accounts/login.html", context)
```

```django
{# accounts/templates/accounts/login.html #}

{% extends 'base.html' %}

{% block content %}
<form action="" method="POST">
  {% csrf_token %}
  {{ form.as_p }}
  <button>로그인</button>
</form>
{% endblock content %}
```
## 6. 로그아웃
```python
# accounts/urls.py

from django.urls import path
from . import views

app_name = "accounts"
urlpatterns = [
    # ...
    path("logout/", views.logout, name="logout"),
]
```

```python
# accounts/views.py

from django.shortcuts import redirect, render
from django.contrib.auth import get_user_model
from django.contrib.auth import login as auth_login
from django.contrib.auth.forms import UserCreationForm
from django.contrib.auth.forms import AuthenticationForm
from django.contrib.auth import logout as auth_logout


def logout(request):
    if request.method == "POST":
        auth_logout(request)
    return redirect("accounts:index")
```
## 7. 회원정보수정
```python
# accounts/urls.py

from django.urls import path
from . import views

app_name = "accounts"
urlpatterns = [
    # ...
    path("update/", views.update, name="update"),
]
```

```python
# accounts/forms.py

from django.contrib.auth.forms import UserChangeForm
from django.contrib.auth import get_user_model


class CustomUserChangeForm(UserChangeForm):
    class Meta:
        model = get_user_model()
        fields = ["username", "email"]

```

```python
# accounts/views.py

from django.shortcuts import redirect, render
from django.contrib.auth import get_user_model
from django.contrib.auth import login as auth_login
from django.contrib.auth.forms import UserCreationForm
from django.contrib.auth.forms import AuthenticationForm
from .forms import CustomUserChangeForm


def update(request):
    if request.method == "POST":
        form = CustomUserChangeForm(request.POST, instance=request.user)
        if form.is_valid():
            form.save()
            return redirect("accounts:index")

    else:
        form = CustomUserChangeForm(instance=request.user)

    context = {
        "form": form,
    }
    return render(request, "accounts/update.html", context)
```

```django
{# accounts/templates/accounts/update.html #}

{% extends 'base.html' %}

{% block content %}
<form action="" method="POST">
  {% csrf_token %}
  {{ form.as_p }}
  <button>회원정보수정</button>
</form>
{% endblock content %}
```
## 8. 회원탈퇴
```python
# accounts/urls.py

from django.urls import path
from . import views

app_name = "accounts"
urlpatterns = [
    # ...
    path("delete/", views.delete, name="delete"),
]
```

```python
# accounts/views.py

from django.shortcuts import redirect, render
from django.contrib.auth import get_user_model
from django.contrib.auth import login as auth_login
from django.contrib.auth.forms import UserCreationForm
from django.contrib.auth.forms import AuthenticationForm
from .forms import CustomUserChangeForm


def delete(request):
    if request.method == "POST":
        user = request.user
        user.delete()
        auth_logout(request)
    return redirect("accounts:index")
```

## 9. 비밀번호 변경
```python
# accounts/urls.py

from django.urls import path
from . import views

app_name = "accounts"
urlpatterns = [
    # ...
    path("password/", views.update_password, name="update_password"),

]
``` 

```python
# accounts/views.py

from django.shortcuts import redirect, render
from django.contrib.auth import get_user_model
from django.contrib.auth import login as auth_login
from django.contrib.auth.forms import UserCreationForm
from django.contrib.auth.forms import AuthenticationForm
from .forms import CustomUserChangeForm
from django.contrib.auth.forms import PasswordChangeForm


def update_password(request):
    if request.method == "POST":
        form = PasswordChangeForm(request.user, request.POST)
        if form.is_valid():
            form.save()
            update_session_auth_hash(request, form.user)
            return redirect("accounts:index")

    else:
        form = PasswordChangeForm(request.user)

    context = {
        "form": form,
    }
    return render(request, "accounts/update_password.html", context)

```

```django
{# accounts/templates/accounts/update_password.html #}

{% extends 'base.html' %}

{% block content %}
<form action="" method="POST">
  {% csrf_token %}
  {{ form.as_p }}
  <button>패스워드변경</button>
</form>
{% endblock content %}
```
