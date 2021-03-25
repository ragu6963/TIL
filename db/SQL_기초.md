### SELECT * FROM table;

특정 `table` 조회

```sqlite
SELECT * FROM users_user;
```

```sqlite
-- 특정 table의 특정 column 만 조회할려면
SELECT rowid, column1 FROM table
```

```sqlite
-- 특정 table의 특정 개수(count)만 조회할려면
SELECT * FROM table LIMIT count;

-- 하나의 row만 조회할려면
SELECT * FROM table LIMIT 1;
```

```sqlite
-- 특정 table의 
SELECT * FROM table LIMIT count OFFSET index;

-- 특정 table의 세번째 값을 조회하할려면
SELECT * FROM table LIMIT 1 OFFSET 2;
```



## CREATE

테이블을 생성

```sqlite
CREATE TABLE classmates(
  name TEXT
);
```



## DROP

특정 테이블 삭제

```sqlite
DROP TABLE table;
```



## INSERT

테이블에 데이터 추가

```sqlite
INSERT INTO table (column1, column2, ...)
	VALUES (value1,value2, ...)
```

```sqlite
-- 지정한 column 순서와 VALUES의 순서가 같아야한다.
INSERT INTO classmates (name,age)
 	VALUES ("홍길동",23); 
```

```sqlite
-- column을 지정안할 때는 순서를 지켜야한다.
INSERT INTO classmates VALUES ("홍길동",23,"서울"); 
```

```sqlite
-- 여러개 INSERT
INSERT INTO classmates 
VALUES ("홍길동",23,"서울"),
("홍길동",23,"서울"),
("홍길동",23,"서울");
```

```sqlite
-- 여러개 INSERT2
INSERT INTO classmates (name)
VALUES ("홍길동"),
("홍길동"),
("홍길동");
```



## NOT NULL

테이블 생성시에 각 필드에 `NOT NULL`  옵션을 준다.

`NOT NULL` 옵션이 있는 필드는 INSERT 할 때 값을 무조건 넣어줘야한다.

```sqlite
CREATE TABLE classmates(
  name TEXT NOT NULL,
  age INT NOT NULL,
  address TEXT NOT NULL
);
```



## WHERE

`조건문`

```sqlite
SELECT * FROM table
WHERE 조건column=value;
```



## DISTINCT

중복없이 column 가져오기

```sqlite
SELECT  DISTINCT column FROM table
WHERE 조건column=value;
```



## DELETE

특정 table에 특정한 row를 삭제할 수 있다.

```sqlite
DELETE FROM table
WHERE 조건column=value;
```



## UPDATE

특정 table의 `row`수정

```sqlite
UPDATE table 
SET column1=value1,column2=value2,...
WHERE 조건column=value;
```



## COUNT

레코드 개수 구하기

```sqlite
SELECT COUNT(*) FROM 테이블;
```



## AVG

평균 계산

```sqlite
SELECT AVG(숫자 자료형 컬럼) FROM 테이블
```



## LIKE

정확한 값 비교가 아닌, 패턴을 확인하여 해당하는 값을 조회

```sqlite
SELECT
	조회컬럼
FROM
	테이블
WHERE
	컬럼1 LIKE 패턴;
```

### 와일드 카드(wild cards)

- _ : 반드시 이 자리에 한 개의 문자가 존재해야한다
- % : 문자가 있을수도 없을수도 있다.

| %    | 2%   | 2로 시작하는 값 |
| ---- | ---- | --------------- |
|      | %2   | 2로 끝나는 값   |
|      | %2%  | 2가 들어가는 값 |

| **_** | _2%               | 아무값이나 들어가고 두번째가 2로 시작하는 값 |
| ---- | ---- | --------------- |
|      | 1_ _ _ _          | 1로 시작하는 4자리인 값                      |
|      | 2 _ % _ % / 2_ _% | 2로 시작하는 최소 3자리인 값                 |



## ORDER

`정렬`

```sqlite
SELECT
   조회컬럼
FROM
   table
ORDER BY
    컬럼1 ASC,  -- 오름차순
    컬럼2 DESC; -- 내림차순
```



## GROUP

특정 COLUMN을 `그룹화`

```sqlite
-- 그룹화
SELECT 컬럼 FROM 테이블 GROUP BY 그룹화할 컬럼;

-- 조건 처리 후에 컬럼 그룹화
SELECT 컬럼 FROM 테이블 WHERE 조건식 GROUP BY 그룹화할 컬럼;

-- 컬럼 그룹화 후에 조건 처리
SELECT 컬럼 FROM 테이블 GROUP BY 그룹화할 컬럼 HAVING 조건식;
```



## ALTER

```sqlite
-- 테이블 이름 바꾸기
ALTER TABLE 이전테이블명
RENAME TO 새로운테이블명;

-- 컬럼 이름 바꾸기
ALTER TABLE 테이블
RENAME COLUMN 이전컬럼명 TO 새로운컬럼명;

-- 컬럼 추가하기
ALTER TABLE 테이블
ADD COLUMN 컬럼명 속성;

-- 컬럼 추가하기 예제1
ALTER TABLE 테이블
ADD COLUMN 컬럼 TEXT;
```

