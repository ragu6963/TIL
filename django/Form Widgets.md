# Form Widgets [공식문서](https://docs.djangoproject.com/en/3.1/ref/forms/widgets/)

> `Widgets`은 HTML `input` 요소의 표현이다. 

### Specifying  widgets

```python
class CommentForm(forms.Form): 
    ###
    comment = forms.CharField(widget=forms.Textarea)
```



### attrs

```python
# input type 및 속성, 클래스 정의
from django import forms
class CommentForm(forms.Form):
    name = forms.CharField(
        widget=forms.TextInput(
            attrs={'size': 10, 
                   'title': 'Your name',
                   'class': 'test test2'
                  }
        )
    )
```

```python
# update 메서드로 정의
from django import forms
class CommentForm(forms.Form):
    name = forms.CharField()
    name.widget.attrs.update({'class': 'test test2'})
    

```

