# [링크](https://www.hackerrank.com/domains/sql?filters%5Bsubdomains%5D%5B%5D=select)

<img src="assets/HacerkRank/1449729804-f21d187d0f-CITY.jpg">

## 1.Revising the Select Query I

Query all columns for all American cities in the **CITY** table with populations larger than `100000`. The **CountryCode** for America is `USA`. 

```sql
SELECT * FROM CITY
    WHERE POPULATION > 100000 and COUNTRYCODE = 'USA'
```

## 2. Revising the Select Query II

Query the **NAME** field for all American cities in the **CITY** table with populations larger than `120000`. The *CountryCode* for America is `USA`. 

```sql
SELECT name FROM CITY
    WHERE population > 120000 and countrycode = "USA"
```

## 3. Select All

Query all columns (attributes) for every row in the **CITY** table.

```sql
SELECT * FROM CITY
```

## 4. Select By ID

Query all columns for a city in **CITY** with the *ID* `1661`.

```sql
SELECT * FROM CITY
    WHERE ID = 1661
```

## 5. Japanese Cities' Attributes

Query all attributes of every Japanese city in the **CITY** table. The **COUNTRYCODE** for Japan is `JPN`.

```sql
SELECT * FROM CITY
    WHERE COUNTRYCODE = "JPN" 
```

## 6. Japanese Cities' Names

Query the names of all the Japanese cities in the **CITY** table. The **COUNTRYCODE** for Japan is `JPN`.

```sql
SELECT name FROM CITY
    WHERE COUNTRYCODE = "JPN"
```

<img src="assets/HacerkRank/1449345840-5f0a551030-Station.jpg">

## 7. Weather Observation Station 1

Query a list of **CITY** and **STATE** from the **STATION** table.

```sql
SELECT CITY, STATE FROM STATION
```

## 8. Weather Observation Station 3

Query a list of **CITY** names from **STATION** for cities that have an even **ID** number. Print the results in any order, but exclude duplicates from the answer. 

```sql
-- 중복없이 ID가 짝수이고, CITY 오름차순
SELECT DISTINCT CITY FROM STATION
    WHERE ID % 2 = 0
    ORDER BY CITY;
```

## 9. Weather Observation Station 4

Find the difference between the total number of **CITY** entries in the table and the number of distinct **CITY** entries in the table.

```sql
-- 개수 - 중복 개수
SELECT count(CITY) - count(DISTINCT CITY) FROM STATION
```

## 10. Weather Observation Station 5

Query the two cities in **STATION** with the shortest and longest *CITY* names, as well as their respective lengths (i.e.: number of characters in the name). If there is more than one smallest or largest city, choose the one that comes first when ordered alphabetically.

