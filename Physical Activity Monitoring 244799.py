#!/usr/bin/env python
# coding: utf-8

# # Physical Activity Monitoring Data: Hypothesis Testing and Predictive Modeling

# Candidate: 244799 <br/>
# Words (excluding references and code): 3274

# ## Introduction

# Wearable technologies which monitor physiology and motion have become increasingly available and popular, benefitting athletes, individuals with certain medical conditions such as heart disease, as well as their use  to improve general physical health and well-being for commercial product consumers.<sup>1</sup> Underlying these technologies are machine learning models which can detect, or predict, which activity a subject is performing based on physiological parameters and motion sensors. Specifically, supervised machine learning methods can be utilised, where models are trained on data where the physical activity or amount of activity is known and labelled, then activity type on future examples can be predicted by the model.
# <br/><br/>
# For this analysis, biological parametric data has been provided by Reiss and Stricker <sup>2</sup> from 9 subjects performing 18 different activities such as cycling, walking and sitting down. These participants wore 3 inertial measurement units (IMUs) as well as a monitor to track their heart rate. Inertial measurements compose of physical disturbances in monitors such as the changing acceleration, magnitude and orientation of technology components during activity. <br/> <br/>
# Using Python programming, the data will be explored and supervised machine learning models will be tested to explore whether this data can be used for predictive purposes. The aims of the work are to highlight parameters from the wearable technologies data that may be able to discriminate between activity types, to show whether activity type can be predicted in these subjects and whether duration of activity can also be predicted based on heart rate data. The results can be of use in future wearable technology engineering development and research. <br/>
# The analysis and testing will be performed in the following stages:
# - Data will be cleaned to ensure good quality data is used in the model. 
# - Exploratory data analysis will be performed to give some indicators on trends and ideas for analysis in the data.
# - A hypothesis test will be carried out between ankle IMUs during walking and lying activities to see if ankle magnitude is significantly different between these parameters.
# - A supervised machine learning classifier will be trained and tested on separate sets of data to and shown to be a good predictor of detecting activity type, thus may work for wearable technology implementation. 
# - A form of dimensionality reduction, Principle Component Analysis, will be applied to the data to see if predictive outcomes can be improved.
# - Linear Regression modelling will be used to predict exercise amount (time duration) for five activities: walking, cycling, jump rope, sitting and lying, using heart rate and time parameters for these activities.
# - Conclusions describe the results in detail and suggest directions for further investigation and improvement.
# <br/><br/>
# 
# The results show that there was significant differences between the ankle magnitude measurements of lying and walking activities, so ankle IMUs, particularly magnitude, may be a good parameter for distinguishing between activities. A random forest classifier proved  highly accurate at making predictions on activity type of the test data subset of the subjects. Dimensionality reduction is possible with this data using PCA, however the reduced dimensions led to extremely marginal improvements in classifier model accuracy, so may not be useful in wearable technology processing practices. Heart rate is shown to be a good predictor of activity amount (seconds) for five of the activities, so may also be a good parameter within wearable technology software.

# ## Data Cleaning

# The data for the 9 subjects, 8 male and 1 female, is imported from the .dat files provided using the Python pandas library, with one file per subject, and then combined into a single data frame. Column names are added as described in the protocol resources from Reiss and Stricker.<sup>2</sup>

# In[1]:


#Import Python packages
import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.utils import check_array
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from mlxtend.evaluate import bias_variance_decomp
from sklearn.decomposition import PCA
from sklearn import metrics
from matplotlib import pyplot as plt
from scipy import stats


# In[2]:


'''
Column names are added 
'''
#New column names
column_names = [
    "timestamp (s)", "activity_no", "heart rate (bpm)", 
    "IMU_hand_temp","IMU_hand_3Dac16_1","IMU_hand_3Dac16_2",
    "IMU_hand_3Dac16_3","IMU_hand_3Dac6_1","IMU_hand_3Dac6_2",
    "IMU_hand_3Dac6_3","IMU_hand_3Dgy_1","IMU_hand_3Dgy_2",
    "IMU_hand_3Dgy_3","IMU_hand_3Dmag_1","IMU_hand_3Dmag_2",
    "IMU_hand_3Dmag_3","IMU_hand_orien_1","IMU_hand_orien_2",
    "IMU_hand_orien_3","IMU_hand_orien_4","IMU_chest_temp",
    "IMU_chest_3Dac16_1","IMU_chest_3Dac16_2","IMU_chest_3Dac16_3",
    "IMU_chest_3Dac6_1","IMU_chest_3Dac6_2","IMU_chest_3Dac6_3",
    "IMU_chest_3Dgy_1","IMU_chest_3Dgy_2","IMU_chest_3Dgy_3",
    "IMU_chest_3Dmag_1","IMU_chest_3Dmag_2","IMU_chest_3Dmag_3",
    "IMU_chest_orien_1","IMU_chest_orien_2","IMU_chest_orien_3",
    "IMU_chest_orien_4","IMU_ankle_temp","IMU_ankle_3Dac16_1",
    "IMU_ankle_3Dac16_2","IMU_ankle_3Dac16_3","IMU_ankle_3Dac6_1",
    "IMU_ankle_3Dac6_2","IMU_ankle_3Dac6_3","IMU_ankle_3Dgy_1",
    "IMU_ankle_3Dgy_2","IMU_ankle_3Dgy_3","IMU_ankle_3Dmag_1",
    "IMU_ankle_3Dmag_2","IMU_ankle_3Dmag_3","IMU_ankle_orien_1",
    "IMU_ankle_orien_2","IMU_ankle_orien_3","IMU_ankle_orien_4",
    "subject"
    ]

#Import data, the 9 subjects are listed in the files 101 - 109
subjects = [x for x in range(101,110)]
dataframes = []
subject_no = 1

for subject in subjects:
    '''
    For every file, the data is converted into a pandas dataframe.
    The data entries in the file are separated by spaces ' '
    An additional column 'subject' is added to keep track of the subjects
    '''
    data_frame = pd.read_csv('subject'+str(subject)+'.dat', delimiter=' ')
    data_frame['subject'] = str(subject_no)
    data_frame.columns = column_names
    dataframes.append(data_frame)
    subject_no += 1 
    
#Join data together
dataframe = pd.concat(dataframes,axis=0,ignore_index=True)


# The duplicate entries of the dataset are removed, to avoid over-influence of certain values and parameters. The numerical columns are also converted into numerical float data types, so they can be used in numerical calculations in Python.

# In[3]:


#Remove duplicate entries
dataframe.drop_duplicates()

#Make a copy of column names
cols_tonumeric = column_names

for label in ['activity_no','subject']: #these columns will remain string type so removed from the list
    if label in cols_tonumeric:
        cols_tonumeric.remove(label)
        
#Convert remaining columns to numerical floats 
for colm in cols_tonumeric:
    dataframe[colm] = dataframe[colm].astype(np.float16)


# The activities in the dataset are labelled by numbers, so an additional column is added for quick identification of activity type. The activity types and numbers are listed in the supporting files from Reiss and Stricker.<sup>2</sup>

# In[4]:


#Create dictionary of number and activity 
activity_IDs = {0:"other",
                1:"lying", 
                2:"sitting", 
                3:"standing", 
                4:"walking", 
                6:"cycling", 
                7:"nordic walking", 
                9:"watching tv", 
                10:"computer work", 
                11:"car driving", 
                12:"ascending stairs",
                13:"descending stairs",
                16:"vacuum cleaning",
                17:"ironing",
                18:"folding laundry",
                19:"house cleaning",
                20:"playing soccer",
                24:"jump rope"
               }

#Map on the activity type dictionary into a new column, based on dictionary keys
dataframe["activity"] = dataframe["activity_no"].map(activity_IDs)


# Since the "other" activity does not provide any valuable insights into activity type, all entries with this activity listed will be removed from the data.

# In[5]:


#Remove all entries from data frame where the activity is "other"
dataframe = dataframe[dataframe.activity != "other"]


# The percentage of missing values for every column in calculated and placed into a table overview. Columns with lots of missing data may be removed, as there may not be enough data to provide a fair insight of the particular parameter in the model. The number of rows and columns of the data are displayed to show how much data there is to use.

# In[6]:


#Create lists to store column names and missing value amount
col_name = []
col_missing_value = []

#Get percentage missing values
for (colname,coldata) in dataframe.iteritems():
    missing_perc = round(dataframe[colname].isna().sum() / len(dataframe) * 100,1)
    col_name.append(colname)
    col_missing_value.append(missing_perc)

#Create data frame and display it 
missing_df = pd.DataFrame([col_missing_value],columns=col_name)
display(missing_df)

#Print shape of the dataset
rows_no, col_no = dataframe.shape
print(f"There are {rows_no} rows and {col_no} columns in the dataset.")


# Most of the columns have relatively low amounts missing data (<0.6%). The heart rate column has a very high percentage of missing data (90.9%), however, since there are almost 2,000,000 data entries, there is still a large amount of heart rate data that can be examined so this column will remain.

# The dataset is split training and test sets, with two thirds for training and one third for testing, as is common practice in developing models.<sup>3</sup> There is an additional subset split of heart rate data that will be used for determining exercise amount, again with a test and training set.

# In[7]:


#Split the dataframe into training and testing set for classification
train_df, test_df = train_test_split(dataframe, 
                                     test_size=0.66, 
                                     random_state=10, 
                                     shuffle=True)

#Subset the data for regression model of heart rates and time 
dataframe_heartrate = dataframe[dataframe["heart rate (bpm)"].notna()]
train_df_heartrate, test_df_heartrate = train_test_split(dataframe_heartrate, 
                                                         test_size=0.66, 
                                                         random_state=10, 
                                                         shuffle=True)


# ## Exploratory Data Analysis

# Exploratory data analysis is performed to provide an overview of general trends in the training data an its parameters. For this, a selection of 15 parameters for 5 activities: lying, sitting, walking, cycling and jump rope are displayed in a series of box plots. This should indicate which parameters may be useful for differentiating between these activities and show some general trends in the dataset.<br/>
# A subset of the data (1000 random entries) is used for the box plot displays within these parameters and activities, so the figures are not overwhelmed with data.

# In[8]:


#Create subset of data with the five actvities 
sub5_train = train_df[train_df["activity"].isin(["lying","sitting","walking","cycling","jump rope"])]

#Get sub sample of this data for EDA 
sub5_train_sample = sub5_train.sample(1000) 

#Plot selection of parameters using the seaborn and matplotlib library
fig, axes = plt.subplots(5, 3, figsize=(12,20))

#Create plot title
plt.suptitle(
    "Boxplot Overviews of Hand, Chest and Ankle Parameters during Five Different Activities",
    fontsize=15)

#Create dictionary for subplot indexes and axis labels
param_index_dict = {
    'IMU_hand_temp':[[0,0],'Hand Temp (C)'],
    'IMU_hand_3Dac16_1':[[0,1],'Hand 3D Acceleration'],
    'IMU_hand_3Dgy_1':[[0,2],'Hand 3D Gyroscope'],
    'IMU_hand_3Dmag_2':[[1,0],'Hand 3D Magnitude'],
    'IMU_hand_orien_1':[[1,1],'Hand Orientation'],
    'IMU_chest_temp':[[1,2],'Chest Temp (C)'],
    'IMU_chest_3Dac16_1':[[2,0],'Chest 3D Acceleration'],
    'IMU_chest_3Dgy_1':[[2,1],'Chest 3D Gyroscope'],
    'IMU_chest_3Dmag_2':[[2,2],'Chest 3D Magnitude'],
    'IMU_chest_orien_1':[[3,0],'Chest 3D Orientation'],
    'IMU_ankle_temp':[[3,1],'Ankle Temp (C)'],
    'IMU_ankle_3Dac16_1':[[3,2],'Ankle 3D Acceleration'],
    'IMU_ankle_3Dgy_1':[[4,0],'Ankle 3D Gyroscope'],
    'IMU_ankle_3Dmag_2':[[4,1],'Ankle 3D Magnitude'],
    'IMU_ankle_orien_1':[[4,2],'Ankle 3D Orientation']   
}

#Function to plot boxplot subplots
def create_subplot(data,column,index,label):
    '''
    This function outputs subplots using the seaborn library and takes:
    data, the column (parameter) to be plotted, the subplot index and the axis label 
    as arguments.
    It outputs a subplot to be displayed.
    '''
    #Create boxplot
    boxplot = sns.boxplot(y=data[column], 
                       x=data['activity'], 
                       data=data,  
                       orient='v',palette="crest", 
                       ax=axes[index[0]][index[1]])
    #Set axis labels
    boxplot.set(xlabel='',
                ylabel=label)
    sns.set(style="dark")
    
#Create subplot for all data in selected
for key, values in param_index_dict.items():
    create_subplot(sub5_train_sample,key,values[0],values[1])

#Show plots
plt.tight_layout()
plt.show()


# From the exploratory data analysis box plot figures, there is a lot of variability visible between means, inter-quartile ranges overall and range of values of the 5 activities for the 15 parameters. This suggests many of them may be good a differentiating between activity type. For example, ankle magnitude looks like it could be a differ between the 5 activities. To explore and expand upon this parameter, the three ankle magnitude parameters will be displayed on a strip plot for lying and walking activities. This will also be done on a sample of 1000 data entries of these activity types. Summary statistics for walking and lying for these three magnitude measures within the same sample will be displayed using the pandas library 'describe' function.

# In[9]:


#Make a dictionary to store the data for walking and lying 
ankle_mag_walklie_data = {}

#Get subset of the data only for lying and walking for all three magnitude measures
walklie_train_data = train_df[train_df['activity'].isin(['lying','walking'])]

walklie_train_data_sample = walklie_train_data.sample(1000)

#Get magnitude measurement 1 data 
walklie_mag1 = walklie_train_data_sample[['activity','IMU_ankle_3Dmag_1']].copy()
walklie_mag1 = walklie_mag1.rename(columns = {'IMU_ankle_3Dmag_1':'ankle magnitude 1'})
walk_mag1 = walklie_mag1[walklie_mag1['activity'].isin(['walking'])]
lie_mag1 = walklie_mag1[walklie_mag1['activity'].isin(['lying'])]
ankle_mag_walklie_data['ankle magnitude 1'] = [walk_mag1,lie_mag1]

#Get magnitude measurement 2 data 
walklie_mag2 = walklie_train_data_sample[['activity','IMU_ankle_3Dmag_2']].copy()
walklie_mag2 = walklie_mag2.rename(columns = {'IMU_ankle_3Dmag_2':'ankle magnitude 2'})
walk_mag2 = walklie_mag2[walklie_mag2['activity'].isin(['walking'])]
lie_mag2 = walklie_mag2[walklie_mag2['activity'].isin(['lying'])]
ankle_mag_walklie_data['ankle magnitude 2'] = [walk_mag2,lie_mag2]

#Get magnitude measurement 3 data 
walklie_mag3 = walklie_train_data_sample[['activity','IMU_ankle_3Dmag_3']].copy()
walklie_mag3 = walklie_mag3.rename(columns = {'IMU_ankle_3Dmag_3':'ankle magnitude 3'})
walk_mag3 = walklie_mag3[walklie_mag3['activity'].isin(['walking'])]
lie_mag3 = walklie_mag3[walklie_mag3['activity'].isin(['lying'])]
ankle_mag_walklie_data['ankle magnitude 3'] = [walk_mag3,lie_mag3]

fig, axs = plt.subplots (1, 3, figsize=(25, 15))

#Set title
plt.title("Sample of Ankle Magnitude Data: Walking vs. Lying",loc='left',size=20)

#Plot ankle magnitude 1 on a strip plot
g1 = sns.stripplot(x="activity",
                   y="ankle magnitude 1",
                   data=walklie_mag1,
                   palette="crest",
                   ax=axs[0])
#Label x and y axis
g1.set_xlabel("Activity",
              fontsize=20)
g1.set_ylabel("IMU Measure Ankle Magnitude (1)",
              fontsize=20)
g1.tick_params(labelsize=15)

#Plot ankle magnitude 2
g2 = sns.stripplot(x="activity",
                   y="ankle magnitude 2",
                   data=walklie_mag2,
                   palette="crest",
                   ax=axs[1])
#Label x and y axis
g2.set_xlabel("Activity",
              fontsize=20)
g2.set_ylabel("IMU Measure Ankle Magnitude (2)",
              fontsize=20)
g2.tick_params(labelsize=15)

#Plot ankle magnitude 3
g3 = sns.stripplot(x="activity",
                   y="ankle magnitude 3",
                   data=walklie_mag3,
                   palette="crest",
                   ax=axs[2])
#Label x and y axis
g3.set_xlabel("Activity",
              fontsize=20)
g3.set_ylabel("IMU Measure Ankle Magnitude (3)",
              fontsize=20)
g3.tick_params(labelsize=15)

#Display plots
fig.tight_layout()
plt.show()

#Show statistical overview of magnitude 2
display(walklie_mag1.groupby('activity').describe().round(2))
display(walklie_mag2.groupby('activity').describe().round(2))
display(walklie_mag3.groupby('activity').describe().round(2))


# All three figures show differences in the distributions and values of the ankle magnitudes of the walking and lying samples from the 9 subjects. There are some clusters which form in both.<br/>
# From the summary statistics, the means of all three magnitude values of lying and walking are different; for <i>ankle magnitude 1</i> -17.86 (lying) and -37.66 (walking), <i>ankle magnitude 2</i> 19.98 (lying) and -0.84 (lying) and <i>ankle magnitude 3</i> -0.46 (lying) and 15.83 (walking). <br/>
# There are also similar amounts of data seen for each activity in the sample (lying 427 and walking 533) which means the comparison between groups is more fair.

# ## Hypothesis Testing

# The ankle magnitude measurements (1,2 and 3) will be further tested to explore whether there is significant difference between ankle magnitude IMUs of lying and walking in the 9 subjects. Significant outcomes would suggest ankle  parameter may be a good measure to differentiate between these activities. For this test, the following hypotheses are proposed:<br/><br/>
# - The Null Hypothesis: There is no significant difference between the magnitude ankle measurements of subjects walking versus subjects lying down.
# - The Alternative Hypothesis: There is a significant difference betweem the magnitude ankle measurements of the subjects walking versus the subjects lying down.
# <br/><br/>
# A two sample t-test will be applied for all three of the ankle magnitude measurements (1,2 and 3) for lying versus walking, to see if a significant p values (<0.05) is seen in the sample.

# This test requires a normal distribution in the parameters where data is evenly spread around the mean, therefore each sample will be tested for normal distribution. This uses the 'normal stats' function form the Python stats library. The output is a probability value indicating normal distribution significance.

# In[10]:


#Test data columns for normal distribution
walk_lie_subsets = []
walk_lie_norm_p_values = []

for key, data in ankle_mag_walklie_data.items():
    #Missing values are removed
    data1 = data[0].iloc[:,1].dropna().tolist()
    data2 = data[1].iloc[:,1].dropna().tolist()
    
    #p values are found for normal distribution
    p1 = stats.normaltest(data1)
    p2 = stats.normaltest(data2)
    
    #results are added to two lists
    walk_lie_subsets.append(key + ' lying')
    walk_lie_norm_p_values.append(p1[1])
    walk_lie_subsets.append(key + ' walking')
    walk_lie_norm_p_values.append(p2[1])

#The outputs are converted into a pandas dataframe
norm_df = pd.DataFrame([walk_lie_norm_p_values],columns=walk_lie_subsets)
display(norm_df.style.hide_index())


# Since all the probability values are close to 0, we can say a there is a very high probability that these parameters are normally distributed. Thus, t testing can be run on the each of the three parameters for ankle magnitude for the lying versus walking data entries.<br/>
# The following code tests each of the three parameters for lying versus walking ankle magnitude difference using the t test function, also from the Python stats package. The output is also a probability value (p value), to indicate whether differences seen between the parameters is likely to have happened by chance. 

# In[11]:


#Dictionary to store P value outcomes
ankle_magnitude_t_tests = {}

#This function gets the p value from the t test by calculating the mean, standard deviation and error
def t_test(sample1,sample2,name):
    '''
    this function calculates a statistical t test of two samples of normal distribution 
    it uses functions from the numpy and stats libraries
    '''
    #Get sample sizes
    sample1_size,sample2_size = len(sample1),len(sample2)
    
    #Get means of samples
    mean1,mean2 = np.mean(sample1),np.mean(sample2)
    
    #Get standard deviations
    std1,std2 = np.std(sample1),np.std(sample2)
    
    #Run the t test
    t_test = stats.ttest_ind_from_stats(mean1,std1,sample1_size,mean2,std2,sample2_size,equal_var=False)
    result = t_test.pvalue/2
    
    return result 

#Iterate through the three magnitude ankle measures 
for i, j in ankle_mag_walklie_data.items():
    sample1, sample2 = j[0].iloc[:,1].dropna().tolist(),j[1].iloc[:,1].dropna().tolist()
    p_value = t_test(sample1,sample2,i)
    ankle_magnitude_t_tests[i]=pd.Series(p_value)

#Display the p values for normal distribution in a table
ankle_mag_ttest_df = pd.DataFrame(ankle_magnitude_t_tests)
display(ankle_mag_ttest_df.style.hide_index())


# The significance values are 0 for all parameters. This suggests there is no, or undetectable, probability that differences between lying and walking magnitudes happened by chance for any of the three parameters. This high significance means we can reject the null hypothesis and accept the alternative hypothesis; that ankle magnitude parameters are significantly different between all three lying and walking parameters of the 9 subjects. <br/>
# Since the difference had such a strong significance, this parameter may be extremely good at differentiating between lying and walking activities and may be subsequently a highly sensitive and good parameter for activity prediction.

# ## Supervised Learning Model: Classification

# Supervised machine learning models are a way of making predictions from parameters of unknown output, based on exposure to labelled datasets, or known outputs. In this context, they can be used to predict, or estimate, which activity is being performed by a subject once the model has been trained on labelled activity data. The training data subset  is used to build the model and then the model is applied to the test set, where labels are removed and the model's predictions are compared to the test set labels to provide an overall accuracy value for the model. A random forest classifier will be used, which is a tree-based classifier. It builds a predictive model by repeatedly applying and testing random tree classifiers to the training data until an optimal tree classifier is found. This method is selected specifically because it provides an easily interpretable metric  predictions success (accuracy), can handle missing data, which has been seen in parameters of this dataset to some degree, and is proven to be efficient on large datasets, so could be very beneficial in future work which may expand to a greater number of individuals and parameters<sup>4</sup>.
# <br/><br/>
# When applying this to the training and test data, the 'activity_no' column is removed as well as the 'timestamp(s)' as the activity detection to be independent of time and any direct label of activity. Lastly, the subjects will removed so predictions are made more generally without subject context. The remaining training data is put into the random forest classifier function from the scikitlearn library to build a model. The model is then applied to the test data and the predictions of the test data activity labels are compared with the actual labels give an accuracy score.

# In[12]:


#Drop some columns from both train and test sets
'''
Three columns are removed which we would not want a model to predict off.
We also remove entries with missing values
'''
train_df_model = train_df.drop(['activity_no', 'timestamp (s)', 'subject'], axis=1).dropna()
test_df_model = test_df.drop(['activity_no', 'timestamp (s)', 'subject'], axis=1).dropna()

#Rename the columns for better reading in visuals
nice_colnames = {
    "IMU_hand_temp":"hand temp",
    "IMU_hand_3Dac16_1":"hand acceleration16 (1)",
    "IMU_hand_3Dac16_2":"hand acceleration16 (2)",
    "IMU_hand_3Dac16_3":"hand acceleration16 (3)",
    "IMU_hand_3Dac6_1":"hand acceleration6 (1)",
    "IMU_hand_3Dac6_2":"hand acceleration6 (2)",
    "IMU_hand_3Dac6_3":"hand acceleration6 (3)",
    "IMU_hand_3Dgy_1":"hand gyroscope (1)",
    "IMU_hand_3Dgy_2":"hand gyroscope (2)",
    "IMU_hand_3Dgy_3":"hand gyroscope (3)",
    "IMU_hand_3Dmag_1":"hand magnitude (1)",
    "IMU_hand_3Dmag_2":"hand magnitude (2)",
    "IMU_hand_3Dmag_3":"hand magnitude (3)",
    "IMU_hand_orien_1":"hand orientation (1)",
    "IMU_hand_orien_2":"hand orientation (2)",
    "IMU_hand_orien_3":"hand orientation (3)",
    "IMU_hand_orien_4":"hand orientation (4)",
    "IMU_chest_temp":"chest temp",
    "IMU_chest_3Dac16_1":"chest acceleration16 (1)",
    "IMU_chest_3Dac16_2":"chest acceleration16 (2)",
    "IMU_chest_3Dac16_3":"chest acceleration16 (3)",
    "IMU_chest_3Dac6_1":"chest acceleration6 (1)",
    "IMU_chest_3Dac6_2":"chest acceleration6 (2)",
    "IMU_chest_3Dac6_3":"chest acceleration6 (3)",
    "IMU_chest_3Dgy_1":"chest gyroscope (1)",
    "IMU_chest_3Dgy_2":"chest gyroscope (2)",
    "IMU_chest_3Dgy_3":"chest gyroscope (3)",
    "IMU_chest_3Dmag_1":"chest magnitude (1)",
    "IMU_chest_3Dmag_2":"chest magnitude (2)",
    "IMU_chest_3Dmag_3":"chest magnitude (3)",
    "IMU_chest_orien_1":"chest orientation (1)",
    "IMU_chest_orien_2":"chest orientation (2)",
    "IMU_chest_orien_3":"chest orientation (3)",
    "IMU_chest_orien_4":"chest orientation (4)",
    "IMU_ankle_temp":"ankle temp",
    "IMU_ankle_3Dac16_1":"ankle acceleration16 (1)",
    "IMU_ankle_3Dac16_2":"ankle acceleration16 (2)",
    "IMU_ankle_3Dac16_3":"ankle acceleration16 (3)",
    "IMU_ankle_3Dac6_1":"ankle acceleration6 (1)",
    "IMU_ankle_3Dac6_2":"ankle acceleration6 (2)",
    "IMU_ankle_3Dac6_3":"ankle acceleration6 (3)",
    "IMU_ankle_3Dgy_1":"ankle gyroscope (1)",
    "IMU_ankle_3Dgy_2":"ankle gyroscope (2)",
    "IMU_ankle_3Dgy_3":"ankle gyroscope (3)",
    "IMU_ankle_3Dmag_1":"ankle magnitude (1)",
    "IMU_ankle_3Dmag_2":"ankle magnitude (2)",
    "IMU_ankle_3Dmag_3":"ankle magnitude (3)",
    "IMU_ankle_orien_1":"ankle orientation (1)",
    "IMU_ankle_orien_2":"ankle orientation (2)",
    "IMU_ankle_orien_3":"ankle orientation (3)",
    "IMU_ankle_orien_4":"ankle orientation (4)",
    "activity":"activity"
    }

#Rename with the nice columns 
train_df_model = train_df_model.rename(columns=nice_colnames)
test_df_model = test_df_model.rename(columns=nice_colnames)
parameters_list = list(train_df_model.columns)

#Split test and training set into x and y variables
train_parameters = train_df_model.drop(['activity'],axis=1)
parameters_list = list(train_parameters.columns)
train_labels = train_df_model['activity']
test_parameters = test_df_model.drop(['activity'],axis=1)
test_labels = test_df_model['activity']

#Create classifier 
'''
Using the scikit learn RandomForestClassifier for supervised machine learning
'''
classifier = RandomForestClassifier(n_estimators=100)

#Fit the model
classifier.fit(train_parameters,train_labels)

#Use the model to make predictors for the test dataset 
test_predictions = classifier.predict(test_parameters)
accuracy1 = round(metrics.accuracy_score(test_labels, test_predictions),5)
accuracy_round = round(accuracy1,3)
print(f"Accuracy: {accuracy_round}")

#Rank the parameter importance
parameter_importance = pd.Series(classifier.feature_importances_,index=parameters_list).sort_values(ascending=False)


# The accuracy of this model is very high at 0.993 (99.3%) accurate, suggesting the model and data are good at activity predictions for the activities in the dataset. It is worth noting that this is only within the context of these nine subjects. The parameter most important in making predictions in the model are shown on a bar chart below.

# In[13]:


#Plot parameter importance
#Specify size of figure
sns.set(rc={'figure.figsize':(15,10)})

#Plot barplot
sns.barplot(x=parameter_importance,
            y=parameter_importance.index,
            palette="crest")

#Add labels
plt.xlabel('Parameter Importance Score',
           size=15)
plt.ylabel('Parameters',
           size=15)
#Add a title
plt.title("Parameter Importance: Wearable Technologies Classifier",
          size=20)

#Set style and size of figure
sns.set_style('dark')
sns.set(rc={'figure.figsize':(15,10)})
plt.show()


# The figure shows that all parameters are useful for predictions, as they all have an importance score above 0. The heart rate (bmp) and hand temperature (C) were shown to be the most important predictor, while chest gyroscope (1) was the poorest predictor. Therefore, if a reduced number of sensors wanted to be determined for use in a predicted model, heart rate and hand or chest temperatures would be advised as the ones to investigate further.

# ## Classification Following Principle Component Analysis

# In a dataset with many parameters, such as the one used in this analysis, rarely all parameters are equally useful in predictions and often some can create noise or reduce the predictive power of better parameters as they may be unrelated to the outcome being examined. Therefore, it can often be useful to reduce the dimensionality of the data before testing. A dimensionality reduction method called Principal Component Analysis (PCA) will be tested to reduce the dimensions representing this dataset. This works by finding a specified number of 'principal components' in the data, or a new parameter which can be a combination of parameters which better fits the data and maintains variation.<sup>5</sup> <br/><br/>
# PCA will be applied to the dataset with varying numbers of principal components (5, 3, 10 and 20) then resulting reduced dimensional data will be re-run with the random forest classifier model to see if reduced dimensionality leads to better prediction accuracy. The accuracy scores of the random forest classifier following PCA using these amounts of components is then plotted along with the original accuracy score of the classifier.

# In[14]:


'''
NOTE: this part takes a few minutes while to run
For this purpose, only a small sample of range of number of components is tested: 3,5,10 and 20
PCA is run on the data for these numbers of principle components, then the classifying model is re-run to give an accruacy score
'''
#Make a list for range of component numbers to test
components_range = [3,5,10,20]

#Create empty list to store the accuracy scores
accuracies = []

for components in components_range:
    #Scale the train and test data
    scaler = StandardScaler()
    train_parameters_scaled = scaler.fit_transform(train_parameters)
    test_parameters_scaled = scaler.transform(test_parameters)
    
    #run PCA for 10 components on train and test data
    pca = PCA(n_components=components)
    pca.fit(train_parameters_scaled)
    train_parameters_scaled_pca = pca.transform(train_parameters_scaled)
    test_parameters_scaled_pca = pca.transform(test_parameters_scaled)
    
    #re-run random forest classifier
    classifier_2 = RandomForestClassifier()
    classifier_2.fit(train_parameters_scaled, train_labels)
    test_predictions_2 = classifier_2.predict(test_parameters_scaled)
    accuracy = metrics.accuracy_score(test_labels, test_predictions_2)
    
    #Add score to the accuracies list
    accuracies.append(accuracy)


# In[15]:


#Plot accuracy for varying range of components
accuracy_df = pd.DataFrame({
                "number of components in PCA":components_range,
                "accuracy score":accuracies
                })
accuracy_df['ones'] = 1

sns.set(rc={'figure.figsize':(10,4)})
sns.set_style('dark')
sns.set_palette('Set2')
#Plot number of principle components versus accuracy score using seaborn line plot
#Create the line plot
acc_plot = sns.lineplot(x="number of components in PCA",
             y="accuracy score",
             hue="ones",
             style="ones",
             markers=True,
             palette="crest",
             data=accuracy_df,
             legend=False)

#Add a line to show the original accuracy score with all parametes
acc_plot.axhline(accuracy1,
                linestyle='--',
                color='seagreen')

#Label the original accuracy line
acc_plot.text(22,
              accuracy1,
              f"original accuracy : {accuracy1}",
             color='seagreen')
#Add a title
plt.title("Varying Number of Principle Components and Subsequent Classifier Accuracy Score",
          size=15)

#Display figure
fig.tight_layout()
plt.show()

accuracy_df_copy = accuracy_df.drop(['ones'], axis=1)
accuracy_df_copy = accuracy_df_copy.rename(columns={'number of components in PCA': 'No. of PCA Components', 'accuracy score': 'Accuracy Score'})
display(accuracy_df_copy.style.hide_index())


# Results from the range of principal components tested in reducing the dimensionality of the data showed all values principal components data improved classifier model accuracy, however this was a very, very marginal improvement (<0.0003) and therefore negligible. Thus, it does not appear that PCA helps improve prediction accuracy within this context. This could be due to the the way PCA outputs maintain the same variance within the data which is exploited by the classifier algorithm. Subsequently, for predictive methods PCA is not recommended for activity prediction in future work.

# ## Supervised Learning Model: Linear Regression for Exercise Amount

# Linear regression is a supervised machine learning method which finds relationships, or gradients, between numerical parameters of a dataset. Using the relationship or gradient found in training data, estimations can be made for an output value of one parameter in the test data, based on known values of the other parameters entered in the model.<sup>6</sup>
# There are various metrics that can be calculated for linear regression models indicating the accuracy of the model on the test data values, or overall wellness of fit of the model with the data:
# - The mean square error (MSE) is the average squares of the errors, or the average difference between predicted and actual values of the test data.
# - The bias describes the difference between the expected value and the true value of the parameter prediction in the training versus the test dataset.
# - The variance indicates consistency, or lack-of consistency, of predictions by the model.
# <br/> <br/>
# Using the subset of data with heart rate values available, a linear regression model will be built and tested to see if exercise amount (duration in seconds) can be predicted for walking, cycling, jump rope, lying and sitting using heart rate (bmp). The MSE, bias and variance will be calculated and given.

# In[16]:


#Function for Linear Regression
def linregres(x_train,y_train,x_test,y_test):
    '''
    This function runs linear regression on test and training set data 
    It takes the data, y value (parameter to be predicted) for both test and training sets as input
    The linear regression function from the scikit learn library is used for the model
    The functions outputs three measurements of model 'goodness': MSE (mean square error), bias and variance
    '''
    regressor = LinearRegression()
    
    #the bias_variance_decomp only accepts numpy arrays as input
    MSE, bias, variance = bias_variance_decomp(regressor, 
                                               x_train, 
                                               y_train, 
                                               x_test,
                                               y_test, 
                                               loss='mse', 
                                               num_rounds=200, 
                                               random_seed=2)
    return MSE, bias, variance

#Split data for cycling, walking and jump rope
heartrate_datasets = {}

#Specify activities to run the linear regression in and get test and training sets 
for activity in ['cycling','walking','jump rope','lying','sitting']:
    train = train_df_heartrate[train_df_heartrate['activity']==activity][['timestamp (s)','heart rate (bpm)']].copy()
    test = test_df_heartrate[test_df_heartrate['activity']==activity][['timestamp (s)','heart rate (bpm)']].copy()
    heartrate_datasets[activity] = [train,test]

#Lists are made to store the output
activity = []
MSEs = []
biases = []
variances = []

for hrset in heartrate_datasets.keys():
    
    #Get subsets of the test and training set per activity 
    hr_train = heartrate_datasets[hrset][0]
    hr_train_y = hr_train['heart rate (bpm)']
    hr_test = heartrate_datasets[hrset][1]
    hr_test_y = hr_test['heart rate (bpm)']
    
    #Convert to numpy arrays 
    hr_train_np = hr_train.to_numpy()
    hr_train_y_np = hr_train_y.to_numpy()
    hr_test_np = hr_test.to_numpy()
    hr_test_y_np = hr_test_y.to_numpy()
    
    #Run linear regression and get outputs
    MSE, bias, variance = linregres(hr_train_np,
                                   hr_train_y_np,
                                    hr_test_np,
                                    hr_test_y_np)
    #Add outputs to lists
    activity.append(hrset)
    MSEs.append(round(MSE,14))
    biases.append(round(bias,14))
    variances.append(round(variance,14))

#Columns for dataframe outcome
linreg_columns = ['Activity','MSE','Bias','Variance']
linreg_df = pd.DataFrame(np.column_stack([activity,MSEs,biases,variances]),
                        columns=linreg_columns)
#Display dataframe
display(linreg_df.style.hide_index())


# The results show for all five activities very low MSE (cycling 3.7x10<sup>-12</sup>, walking 1.8x10<sup>-11</sup>, jump rope 4.7x10<sup>-11</sup>, lying 1.6x10<sup>-11</sup>, sitting 1.3x10<sup>-11</sup>) bias (cycling 5.6x10<sup>-13</sup>, walking 2.2x10<sup>-12</sup>, jump rope 8.1x10<sup>-12</sup>, lying 6.4x10<sup>-13</sup>, sitting 1.4x10<sup>-13</sup>) and variance values (cycling 3.1x10<sup>-12</sup>, walking 1.5x10<sup>-11</sup>, jump rope 3.9x10<sup>-11</sup>, lying 1.5x10<sup>-11</sup>, sitting 1.2x10<sup>-11</sup>). This suggests within this data, a linear regression model is very accurate at estimating the amounts of cycling, walking, jump roping, lying or sitting (duration in seconds) for these 9 subjects. A very low variance, however, suggests the model may be an extremely close fit to heart rate values for these subjects and therefore be over-fitting. This will likely mean that when the same model is applied to new data for different subjects or other environmental variations, it may lead to poorer predictions and accuracy. 

# ## Conclusion

# ### Main Findings

# The analysis demonstrated some good predictive outcomes of various models and indicators for further analysis of IMU parameters of a range of activities from the dataset provided. The main outcomes showed:
# - Exploratory data analysis overviewed trends in the data for 5 activities, many of which could be further investigated in future work as there are lots of differences seen within the means and ranges of these five activities.
# - Hypothesis testing on ankle magnitude measurements showed very significant differences between lying and walking, suggesting this parameter is good for discriminating between these activities. This may apply to other activities for effective discrimination, but further hypothesis testing should be performed to confirm this. 
# - Random forest supervised learning was shown to be a highly accurate model for predicting subject activity in unseen data of these 9 subjects with 99.3% predictive accuracy. This is a very promising results but should be cautioned due to the huge amount of data supplied with relatively few subjects of similar demographics.
# - Reducing data dimensionality via PCA methods is possible with this data, however it did not improve any notable accuracy for the random forest classifier accuracy for predicting activity type.
# - Predicting activity duration of five activities: cycling, walking, jump rope, lying and sitting, based on heart rate (bpm) using linear regression lead to a very accurate (low mean square error) and low-bias model. However, since the variance was low, using this model on new subjects should be done with caution, as it is a possible over-fit to the activities on these nine subjects. 

# ### Limitations

# As briefly mentioned, a key limitation of the analysis is the limited number of subjects used (9) and their demographics. 8 of the subjects were male and only 1 was female, 8 out of the 9 subjects were right hand dominant and all 9 were between the ages of 23 and 32, which is a very limited age range. Future work should include a more diverse pool of subjects in terms of age, gender and physical ability. This may result in a regression model with a slightly higher variance, but also one that may less of an over-fit and generalise better in unseen data from new subjects, thus be more useful in wearable technology software development. Similarly, the 99.3% accuracy seen in the classifier model may also reflect the huge amount of data used and tested for on the same 9 subjects all with similar demographics. This also demands re-modelling and training with a wider pool of subjects to work better in general software applications.
# <br/> <br/>
# PCA methods were applied but did not improve accuracy of the classifier. Despite this failure, there may still be a need for parameter reduction since there remains a large amount of parameters used on various IMUs across the subjects body. Since wearable technologies often need to be small, compact and applied to a singular site on the body, further investigations can be done to refine which body position parameters work best within these predictive models.
# <br/> <br/>
# Lastly, the models could and should be re-tested again on randomly re-sample data or bootstrapping methods to confirm accuracy and signifance measures. This would increase reliability and confidence in these results.

# ### Future Work

# This analysis has proven the suitability of supervised machine learning methods, random forest classifiers and linear regression, in wearable technology data where physiological and IMU parameters can be used to produce predictions with extremely high accuracy. The data collection and analytical methods shown should be used again but in a wider pool of subjects. Future ventures should increasingly focus on the best positioning on IMUs and finding models which generalise to unseen data of new subjects rather than those also in the training set.

# ## References

# 1. Vijayan, Vini, James P. Connolly, Joan Condell, Nigel McKelvey, and Philip Gardiner. "Review of Wearable Devices and Data Collection Considerations for Connected Health." Sensors 21, no. 16 (2021): 5589.
# 2. A. Reiss and D. Stricker. Introducing a New Benchmarked Dataset for Activity Monitoring. The 16th IEEE International Symposium on Wearable Computers (ISWC), 2012.
# 3. R for Medicine and Biology. United States: Jones & Bartlett Learning, 2009.
# 4. Pal, Mahesh. "Random forest classifier for remote sensing classification." International journal of remote sensing 26, no. 1 (2005): 217-222.
# 5. Jolliffe, Ian. "Principal component analysis." Encyclopedia of statistics in behavioral science (2005).
# 6. Uyanık, Gülden Kaya, and Neşe Güler. "A study on multiple linear regression analysis." Procedia-Social and Behavioral Sciences 106 (2013): 234-240.

# In[ ]:





# In[ ]:




