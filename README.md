# hiking_sentiment_analysis
Using data from WTA trip reports, review the reports, do sentiment analysis, and pair with hiking characteristics. Then recommend future hikes per user


Reports page: https://www.wta.org/go-outside/trip-reports

Guide: https://realpython.com/beautiful-soup-web-scraper-python/
Examples: https://brightdata.com/blog/how-tos/beautiful-soup-web-scraping

Summary of Process, Findings and Potential Application:

Process:
-Web Scrape:
-The process started by scraping various necessary data points from the hiking website, using Selenium and Beautiful Soup. 
-This was an initial challenge, as the only times I have done web scrpaing have been done on very simply structured web pages or things like Twitter, which are very easy to get lots of data from. There was a lot of odd customizing and crash-avoidance that needed to be trial-and-error built into the code to consistently get good data, from all pages.

-Cleaning:
-After scraping, the process then turned to cleaning the data, which was of course a mess. Sorting the column types, and cleaning up the numeric columns, to be integers, dates, floats, etc. , along with visualizing the dataset to check for outliers and other potential weakenesses. 
-Then doing the categorical cleaning, and encoding, enabled me to do some simple sentiment analysis. This was crucial, as the signal for weather something was a good recommendation was based on this signal. This took some trial-and-error as well, but I landed on text blob, for ease of use. 
-From there I did some standard imputation, using KNN, which I thought to be a good fit for this dataset, its structure and nature.
-After outlier handling and standardization, I moved onto setting up PCA. I believed PCA would be a potentially valuable way to increase accuracy, if initial models failed.

-Modeling: 
-Using a standard Collaborative Filtering approach using KNN yielded good results, and with very minor tuning I arrived at 95% average accuracy and precision. Utilizing the PCA set didn't yield appreciatable performance improvement. However, for future research I could do more tuning of the PCA ,as well general feature selection to help decrease noise in the model.
-Another potential future point of research would be to integrate the time factor. Different hikes are naturally better at different times of year, so it would be worth while to update this as a feature of the input, so as to provide more resilient recommendations. 

-I also spent time building and tuning a Neural Collaborative Filtering model, to see if I could get even better performance on a consistent basis, and...

Findings:
-The initial KNN model provided statisfactory results (accuracy and precision), and could be utilized for initial easy results. The PCA approach didn't yield expected increase in performance, despite having many encoded columns. The Neural Network.... There is room for future improvement, and implementation of other features. 


Potential Application:
-This model could be patched in as the back end recommendation agent, perhaps where each time someone leaves a review of one hike, the model could refer another one for them, increasing time on the platform.
-Or if you had a travel/tour website, that made money off of ad revenue and recommending excursions, this simple add on could be used to attract users to the platform, by recommending them consistently good hikes. 
-Knowing the hiking community, people are consistently looking to spend more time out doors, and enjoy the sense of community they get with giving reviews and recieving personal referals, and this could be a further mechanism to encourage that process. Additionally, with the other features I recommended for above for further research, the models accuracy could improve, and depending on the load of users, could be scaled easily, especially with the NN, into a cloud based platform. 


Learning:
-I learned a lot from this project, about how to do more complex web scraping and sentiment analysis, to utilizing a new Neural Network. I hope this project demonstrates my ability to problem solve, and overcome unanticipated challenges. As well, it shows an overlap of my passions, data science and the outdoors, as a creative way for me to portray my self starter attitude, and natural creativity when it comes to coding. 

