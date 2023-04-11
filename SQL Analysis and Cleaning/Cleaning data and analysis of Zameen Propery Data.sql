use Projects;

select * from [zameen property data updated]
order by property_id asc;

--Delete the useless columns
alter table [zameen property data updated]
drop column page_url

--Get only date from date column
select date_added
from [zameen property data updated]

select CONVERT(date,date_added)
from [zameen property data updated]

update [zameen property data updated]
set date_added=CONVERT(date,date_added) --This one is not working

alter table [zameen property data updated]
add Date_Updated date

update [zameen property data updated]
set Date_added=CONVERT(date,date_added)

select * from [zameen property data updated]

--Check for null values

select * from [zameen property data updated]
where agency is null

select distinct(agent) from [zameen property data updated]
where agent='%self%'

select ISNULL(agency,'Self')
from [zameen property data updated]

update [zameen property data updated]
set agency=ISNULL(agency,'self')

select agency from [zameen property data updated]
where agency is null

			--Tackle the agent column

select agent from [zameen property data updated]
where agent is not null

select agent from [zameen property data updated]
where agent = 'Self'

select ISNULL(agent,'Self')
from [zameen property data updated]

update [zameen property data updated]
set agent=ISNULL(agent,'Self')

--Check other null values if any
select * from [zameen property data updated]
where location_id is null

select * from [zameen property data updated]
where property_type is null

select * from [zameen property data updated]
where price is null

select * from [zameen property data updated]
where location is null

select * from [zameen property data updated]
where city is null

select * from [zameen property data updated]
where province_name is null

select * from [zameen property data updated]
where latitude is null or longitude is null or baths is null or area is null or purpose is null or
bedrooms is null or date_added is null or [Area Type] is null or [Area_size_in_marla] is null or [Area Category] is null or
date_added is null
--Null values are not any more there

select * from [zameen property data updated]

--Covert Kanals into Marla

select a.[Area Type],a.[Area_size_in_marla],a.[Area_size_in_marla]*20 as AreaInMarla
from [zameen property data updated] a
join [zameen property data updated] b
on a.property_id=b.property_id
where a.[Area Type]<>'Marla';

alter table [zameen property data updated]
add AreaInMarla float

update a
set a.[Area_size_in_marla]=a.[Area_size_in_marla]*20
from [zameen property data updated] a
join [zameen property data updated] b
on a.property_id=b.property_id
where a.[Area Type]<>'Marla';

select * from [zameen property data updated]

--Drop useless columns
alter table [zameen property data updated]
drop column area,date_added,[Area Type],[Area Category],AreaInMarla

select * from [zameen property data updated]


--separate location from city
select 
replace(location,'Bahria Town Karachi','Bahira Town'),
replace(location,'Dream Avenue Lahore','Dream Avenue')
from [zameen property data updated]

update [zameen property data updated]
set location=replace(location,'Bahria Town Karachi','Bahira Town')

update [zameen property data updated]
set location=replace(location,'Dream Avenue Lahore','Dream Avenue')

update [zameen property data updated]
set location=replace(location,'Kingdom Valley Islamabad','Kingdom Valley')

update [zameen property data updated]
set location=replace(location,'Bahria Town Rawalpindi','Bahria Town')


select * from [zameen property data updated]


select distinct(location) from [zameen property data updated]
where location like '%Rawalpindi%'


select distinct(city) from [zameen property data updated]


select 
PARSENAME(replace(location,' ','.'),2),
PARSENAME(replace(location,' ','.'),3)
from [zameen property data updated]


alter table [zameen property data updated]
drop column lastlocation
