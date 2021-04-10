# ManyToMany Relationship [문서](https://docs.djangoproject.com/en/3.1/topics/db/examples/many_to_many/)

Django 에서 `N:M` 관계를 정의 하기위한 방법

`ManyToManyField` 를 통해서 `N:M` 관계를 정의한다.

### N:M 관계란?

`1:N`관계는 N이 1을 참조하는 관계라면 `N:M`은 서로가 서로를 참조하는 관계이다.

즉, 서로가 서로를 `1:N` 관계로 보고 있는 것이다. 예를들면, `의사`와 `환자`은 N:M 관계이다.

### ManyToManyField()

Django에서 두 Model을 `N:M`관계로 만들어주는 필드

두 Model 중 하나의 모델에서만 정의한다.

#### 사용예시

```python
class Doctor(models.Model):
    name = models.CharField(max_length=30)
    
class Patient(models.Model):
    name = models.CharField(max_length=30)
    doctors = models.ManyToManyField(Doctor)
```

#### 중계 테이블

`ManyToManyField`를 사용하면 Django에서 자동으로 중계테이블을 생성해준다.

> sqlmigrate 실행 결과

```sql
BEGIN;
--
-- Create model Doctor
--
CREATE TABLE "crud_doctor" ("id" integer NOT NULL PRIMARY KEY AUTOINCREMENT, "name" varchar(30) NOT NULL);
--
-- Create model Patient
--
CREATE TABLE "crud_patient" ("id" integer NOT NULL PRIMARY KEY AUTOINCREMENT, "name" varchar(30) NOT NULL);
--
-- Create crud_patient_doctors : 중계 테이블 생성
-- 
CREATE TABLE "crud_patient_doctors" ("id" integer NOT NULL PRIMARY KEY AUTOINCREMENT, "patient_id" integer NOT NULL REFERENCES "crud_patient" ("id") DEFERRABLE INITIALLY DEFERRED, "doctor_id" integer NOT NULL REFERENCES "crud_doctor" ("id") DEFERRABLE INITIALLY DEFERRED);
CREATE UNIQUE INDEX "crud_patient_doctors_patient_id_doctor_id_a9b37ff1_uniq" ON "crud_patient_doctors" ("patient_id", "doctor_id");
CREATE INDEX "crud_patient_doctors_patient_id_d5c57f03" ON "crud_patient_doctors" ("patient_id");
CREATE INDEX "crud_patient_doctors_doctor_id_f06fef52" ON "crud_patient_doctors" ("doctor_id");
COMMIT;
```

Model을 정의한 Doctor과 Patient뿐만 아니라 중계테이블인 `patient_doctors`도 생성한다.

---

### related_name

- 위 코드에서 `Patient`가 `Doctor`를 참조할 때는 필드명인 `patient.doctors`를 사용하면 된다.
- 위 코드에서 `Doctor`가 `Patient`를 역참조할 때는 `doctor.patient_set`을 사용해야한다.

이를 변경할려면  `related_name` 속성으로 역참조할 때 변수명을 변경할 수 있다.

```python
class Doctor(models.Model):
    name = models.CharField(max_length=30)
    
class Patient(models.Model):
    name = models.CharField(max_length=30)
    doctors = models.ManyToManyField(Doctor, related_name="patients")
```

이제 `Doctor`가 `Patient`를 역참조할 때 `doctor.patients`로 참조할 수 있다.

```python
# Patient 가 Doctor을 참조
patient1.doctors.all()

# Doctor 가 Patient를 역참조
doctor1.patient.all()
```

---

### Related Manager(add, remove)

관계를 맺고`add`, 없에는`remove` 메서드

> add는 중복된 관계는 만들지 않는다. 

```python
# patinet1과 doctor1의 관계를 만든다
patient1.doctors.add(doctor1)

# doctor1와 patient2의 관계를 만든다
doctor1.patients.add(patient2)

# doctor1와 patient1의 관계를 없엔다
doctor1.patients.remove(patient1)
```

---

### symmetrical(대칭적)

모델이 자신`self`을 참조할 때`재귀참조` 한 방향으로만 참조가 이루어지고 하고 싶을 때  `symmetrical = False` 를 사용한다.

팔로우를 할 때 한 방향으로만 팔로우가 이루어 지는데 `symmetrical = False` 가 이러한 역할을 한다.

```python
class People(models.Model):
    name = models.CharField(max_length=30)
    follow = models.ManyToManyField("self", symmetrical=False)
```

```python
people1 = People()
people1.name = "유저1"
people1.save()
people2 = People()
people2.name="유저2"
people2.save()
people2.people_set.add(people1)

# 한쪽에만 관계가 생겼다.
people2.people_set.all()
# <QuerySet [<People: People object (1)>]>
people1.people_set.all()
# <QuerySet []>
```

