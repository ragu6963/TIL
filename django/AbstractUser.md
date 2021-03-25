# [Customizing authentication in Django](https://docs.djangoproject.com/en/3.1/topics/auth/customizing/#customizing-authentication-in-django)

## 1. 커스텀 Model

```python
# accounts/models.py

from django.db import models
from django.contrib.auth.models import AbstractUser


class User(AbstractUser):
    # 확장할 필드 추가
    nation = models.CharField(default="korea", max_length=50)
```
## 2. 설정

```python
# settings.py

# AUTH_USER_MODEL = "App.Model"
AUTH_USER_MODEL = "accounts.User"
```

## 3. 관리자 페이지 등록

> `AbstractUser` 를 사용해서 User를 커스텀하면 관리자페이지에 등록을 해줘야한다.

```python
# accounts/admin.py

from django.contrib import admin
from .models import User


admin.site.register(User)
```

## 4. 회원가입

> `UserCreationForm`을 상속받은 새로운 Form을 만들어 사용하는 것을 제외하고는 기존 방식과  동일하다.

```python
# accounts/forms.py

from django.contrib.auth.forms import UserCreationForm


# UserCreationForm 상속
class CustomUserCreateForm(UserCreationForm):
    
    class Meta(UserCreationForm.Meta):
        model = get_user_model()
        # fields = UserCreationForm.Meta.fields + ('custom_field',)
        fields = UserCreationForm.Meta.fields + ("nation",)
```

```python
# accounts/views.py

from django.shortcuts import redirect, render
from django.contrib.auth import login as auth_login
from .forms import CustomUserCreateForm


def signup(request):
    if request.method == "POST":
        # CustomUserCreateForm 사용
        form = CustomUserCreateForm(request.POST)
        if form.is_valid():
            user = form.save()
            auth_login(request, user)
            return redirect("accounts:index")

    else:
        form = CustomUserCreateForm

    context = {
        "form": form,
    }
    return render(request, "accounts/signup.html", context)
```
