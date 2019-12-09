### 1. Big Data Problem
###### Improving efficiency in public transportation by adjusting traffic lights based on big data analysis.
#### 1.1 Objectives
###### a. Analyzing the traffic situations in the current and nearby road.
###### b. Predicting the effects of traffic lights’ time interval on the congestions and traffic accidents using some variables like whether, peak hours, current amounts of vehicles and so on, then adjusting traffic lights according to it.
###### c. Avoiding the extreme long or short interval of traffic lights.
#### 1.2 Big data properties
###### a. Volume: There are millions of vehicles in cities, especially developed cities, we can consider it as a high volume.
###### b. Velocity: The real time traffic situation is captured by cameras above the roads, which have a high frequency, so it can give us real time information with high velocity.
###### c. Variety: Except for amounts of vehicles, variables like whether, time, speed of vehicles also needed to be considered in the process of analysis.
### 2. Workflow
###### The workflow of this analysis can be similar with the workflow of high frequency trading.
###### Firstly, we can get information from cameras, local meteorological bureau, and news (to check whether there is a traffic accident nearby).
###### Then, we put these data in an ETL (Extraction, Transformation, Loading) system to extract useful data and transform them into well-structured data. At the same time, we need to store the data in ODS (Operational Data Store).
###### In the third step, we can do the calculation in an engine and figure out the optimal time intervals of traffic lights. After the analysis, the engine will pass the command to the system controlling the traffic lights.
###### Finally, we can check whether the adjustment is useful or whether it is inefficient, in this way, we can update our model to adjust the traffic situation better.
### 3. Database to be used
###### We can use MongoDB to store the data we need for 2 reasons:
###### a. Obviously the data we used to analyze optimal time interval of traffic lights is structured so that we need to use NoSQL database instead of SQL database.
###### b. Among NoSQL database, MongoDB is suitable to handle problems of documentation. In this case, we need to match a certain road with many characteristics such as traffic flow and speed, drainage situations in rainy days, etc.
