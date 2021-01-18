# 단축평가(short-circuit evalution)

>첫 번째 값이 확실할 때, 두 번째 값은 확인하지 않는다.
>
>뒷쪽의 연산자를 판단하지 않아도 되기 때문에 속도가 향상

### Example

```python
# and

print(False and True)  # 결과 False
print(False and False) # 결과 False
# 첫번째 값이 False 이므로 두 번째 값은 확인하지 않고 False로 값이 결정된다.

# and 는 첫 번째 값이 True 라면 두 번째 값이 결과로 나온다.
('a' and 'b') in 'aeiou' # 결과 False
# 'a' and 'b' => 첫 번째 값이 True 이기 때문에 'b'가 값으로 결정된다.
# 'b' in 'aeiou' => False
```

```python
# or

print(True or False) # 결과 True
print(True or True)  # 결과 True
# 첫번째 값이 True 이므로 두 번째 값은 확인하지 않고 True 값이 결정된다.

# or 은 첫 번째 값이 True 라면 첫 번째 값이 결과로 나온다.
('a' or 'b') in 'aeiou' # 결과 True
# 'a' or 'b' => 첫 번째 값이 True 이기 때문에 'a'가 값으로 결정된다.
# 'a' in 'aeiou' => True
```



