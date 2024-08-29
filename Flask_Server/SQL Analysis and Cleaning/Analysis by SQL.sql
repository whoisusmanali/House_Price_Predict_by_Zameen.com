use Projects

select * from [zameen property data updated]

select distinct(purpose) from [zameen property data updated]
select distinct(property_type) from [zameen property data updated]

--average sale price of houses in karachi
select property_type,avg(price) as price, location,city,avg(baths) as baths,avg(bedrooms) as bedrooms,avg(Area_size_in_marla) as area
from [zameen property data updated]
where property_type='House' and city='Karachi' and purpose='For Sale'
group by property_type,location,city
order by price desc


--average rent price of houses in karachi
select property_type,avg(price) as price, location,city,avg(baths) as baths,avg(bedrooms) as bedrooms,avg(Area_size_in_marla) as area
from [zameen property data updated]
where property_type='House' and city='Karachi' and purpose='For Rent'
group by property_type,location,city
order by price desc


--What type of property is more expensive to buy in Karachi
select property_type,avg(price) as Average_Price,AVG(Area_size_in_marla) as Average_Area
from [zameen property data updated]
where purpose='For Sale' and city='Karachi'
group by property_type
--By size
select property_type,avg(price) as Average_Price,AVG(Area_size_in_marla) as 'Average_Area(Marla)'
from [zameen property data updated]
where purpose='For Sale' and city='Karachi'
group by property_type,Area_size_in_marla
order by property_type --Prices may differ because of different location


--What type of Property is more expensive to rent in karachi
select property_type,avg(price) as Average_Price,AVG(Area_size_in_marla) as 'Average_Area(Marla)'
from [zameen property data updated]
where purpose='For Rent' and city='Karachi'
group by property_type,Area_size_in_marla
order by property_type --Prices are different becuase of location

--Without Size
select property_type,avg(price) as Average_Price
from [zameen property data updated]
where purpose='For Rent' and city='Karachi'
group by property_type	--Prices are different becuase of location


--Which location is more expensive in karachi to rent a property
select location,avg(price)
from [zameen property data updated]
where city='Karachi' and purpose='For Rent'
group by location
order by avg(price) desc

--Which City is more expensive to buy property by per marla
select city,sum(price)/sum(Area_size_in_marla) as Per_Marla_Price
from [zameen property data updated]
where purpose='For Sale'
group by city
order by Per_Marla_Price desc    --So Karachi is more expensive to buy property in Pakistan

--Which location in Islamabad is more expensive to buy 
select location,sum(price)/sum(Area_size_in_marla) as Per_Marla_Price
from [zameen property data updated]
where city='Islamabad' and purpose='For Sale'
group by location
order by Per_Marla_Price desc


--Which agency contribute more in property ads
select agency,count(property_type) as Number_of_Ads
from [zameen property data updated]
where agency !='self'
group by agency
order by count(property_type) desc

select distinct(agency) from [zameen property data updated]

--Top 10 Agents who contibuted more in ads
select distinct(agent) from [zameen property data updated]

select top 10 agent,count(property_type) as Number_of_Ads
from [zameen property data updated]
where agency !='Self'
group by agent
order by count(property_type) desc


--Percentage of how many people want to sale or rent by propery type
select property_type,city,count(location) as Total_Ads
from [zameen property data updated]
group by property_type,city
order by city

--Creating view by Agency Contribution
create view AdsByAgency as 
select agency,count(property_type) as Number_of_Ads
from [zameen property data updated]
where agency !='self'
group by agency
--order by count(property_type) desc

--which City put more Area in Ads
select city,sum(Area_size_in_marla) as Total_Area
from [zameen property data updated]
where purpose='For Sale'
group by city
order by Total_Area desc  --Mostly Lahori want to sale their propery

--Who many people want to rent their propery
select city,sum(Area_size_in_marla) as Total_Area
from [zameen property data updated]
where purpose='For Rent'
group by city
order by Total_Area desc    --Mostly Islamabadis want to rent their property

