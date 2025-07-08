#!/usr/bin/env python
# coding: utf-8

# In[90]:


# Initialise SQL
get_ipython().run_line_magic('load_ext', 'sql')
import sqlite3
from sqlalchemy import create_engine

# Data manipulation
import pandas as pd           # pd is the standard alias for pandas
import numpy as np            # np is the standard alias for numpy
from scipy.stats import spearmanr

# Data visualization
import matplotlib.pyplot as plt  # plt is the alias for matplotlib's pyplot
import seaborn as sns             # sns is the alias for seaborn

# Machine learning
import sklearn                   # Main scikit-learn package
from sklearn.model_selection import train_test_split  # Split data into train/test sets
from sklearn.preprocessing import StandardScaler       # Feature scaling
from sklearn.linear_model import LinearRegression       # Example model
from sklearn.model_selection import train_test_split 
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer


# In[64]:


file_path = r"C:\Users\Julie\Desktop\London Weather\London Weather EDA\CleanLWDS.csv"
df = pd.read_csv(file_path)


# In[65]:


df.head()


# In[13]:


df.tail()


# In[14]:


duplicates = df[df.duplicated()]

# Number of duplicate rows
num_duplicates = duplicates.shape[0]

print(f"Number of duplicate rows: {num_duplicates}")
print("Duplicate rows:")
print(duplicates)


# In[15]:


null_counts = df.isnull().sum() #description of null values per column
print(null_counts)


# In[16]:


print(df.dtypes) #check data types


# In[9]:


#df['cloud_cover'] = df['cloud_cover'].astype(int) #Change Cloud Cover to Integer it can only 1 2 .. 7 8. 


# In[13]:


# Boxplot: Sunshine grouped by cloud_cover
sns.boxplot(x='cloud_cover', y='sunshine', data=df)
plt.title('Sunshine Hours by Cloud Cover (Okta)')
plt.show()

# Calculate Spearman correlation
corr, p_val = spearmanr(df['cloud_cover'], df['sunshine'])
print(f'Spearman correlation: {corr:.3f}, p-value: {p_val:.3g}')

# Mean sunshine per cloud_cover
mean_sunshine = df.groupby('cloud_cover')['sunshine'].mean()
print(mean_sunshine)


# In[14]:


# Select rows where cloud_cover == 9
cloud_cover_9_records = df[df['cloud_cover'] == 9]

# Show the records
print(cloud_cover_9_records)

# Optional: count how many such records exist
count_9 = cloud_cover_9_records.shape[0]
print(f'Number of records with cloud_cover = 9: {count_9}')


# In[93]:


# Calculate average sunshine for each cloud_cover level
avg_sunshine = df.groupby('cloud_cover')['sunshine'].mean().reset_index()

# Scatter plot
plt.scatter(avg_sunshine['cloud_cover'], avg_sunshine['sunshine'])
plt.xlabel('Cloud Cover (oktas)')
plt.ylabel('Average Sunshine (hours)')
plt.title('Scatter Plot: Cloud Cover vs Average Sunshine')
plt.grid(True)
plt.show()


# In[17]:


# 1. Calculate average sunshine per okta (cloud_cover 0 to 8)
avg_sunshine = df[df['cloud_cover'] != 9].groupby('cloud_cover')['sunshine'].mean()

# 2. Define a function to map sunshine to closest cloud_cover based on avg_sunshine
def map_sunshine_to_okta(sunshine_value):
    # Calculate absolute difference with each average sunshine
    diffs = np.abs(avg_sunshine - sunshine_value)
    # Return the okta (cloud_cover) with the minimum difference
    return diffs.idxmin()

# 3. Apply this function only to rows where cloud_cover == 9
df.loc[df['cloud_cover'] == 9, 'cloud_cover'] = df.loc[df['cloud_cover'] == 9, 'sunshine'].apply(map_sunshine_to_okta)

# Now all 9s are replaced with the closest okta based on sunshine hours


# In[18]:


# Calculate average sunshine for each cloud_cover level
avg_sunshine = df.groupby('cloud_cover')['sunshine'].mean().reset_index()

# Scatter plot
plt.scatter(avg_sunshine['cloud_cover'], avg_sunshine['sunshine'])
plt.xlabel('Cloud Cover (oktas)')
plt.ylabel('Average Sunshine (hours)')
plt.title('Scatter Plot: Cloud Cover vs Average Sunshine')
plt.grid(True)
plt.show()


# In[94]:


null_counts = df.isnull().sum() #description of null values per column
print(null_counts)


# In[20]:


# Calculate average sunshine per valid cloud_cover (excluding NaNs)
avg_sunshine = df[df['cloud_cover'].notna()].groupby('cloud_cover')['sunshine'].mean()

# Function to find the closest cloud_cover by sunshine value
def estimate_cloud_cover(sunshine_val):
    diffs = np.abs(avg_sunshine - sunshine_val)
    return diffs.idxmin()

# Find rows where cloud_cover is missing
missing_mask = df['cloud_cover'].isna()

# Apply the estimation only for missing cloud_cover records
df.loc[missing_mask, 'cloud_cover'] = df.loc[missing_mask, 'sunshine'].apply(estimate_cloud_cover)


# In[21]:


null_counts = df.isnull().sum() #description of null values per column
print(null_counts)


# In[22]:


df['cloud_cover'] = df['cloud_cover'].astype(int) #converting cloud_cover data type to integer, now possible. 


# In[24]:


plt.figure(figsize=(8,5))
sns.scatterplot(data=df, x='sunshine', y='global_radiation', alpha=0.6)
sns.regplot(data=df, x='sunshine', y='global_radiation', scatter=False, color='red')
plt.title('Global Radiation vs Sunshine Hours')
plt.xlabel('Sunshine Hours')
plt.ylabel('Global Radiation')
plt.show()


# In[25]:


correlation = df['global_radiation'].corr(df['sunshine'])
print(f"Pearson correlation between global radiation and sunshine: {correlation:.3f}")


# In[26]:


df['sunshine_bin'] = pd.cut(df['sunshine'], bins=range(0, int(df['sunshine'].max()) + 2, 2))
grouped = df.groupby('sunshine_bin')['global_radiation'].mean().reset_index()

sns.barplot(data=grouped, x='sunshine_bin', y='global_radiation')
plt.xticks(rotation=45)
plt.title('Average Global Radiation by Sunshine Hours Bin')
plt.xlabel('Sunshine Hours Bin')
plt.ylabel('Average Global Radiation')
plt.show()


# In[27]:


max_sunshine = df['sunshine'].max()
print(f"Max sunshine hours: {max_sunshine}")


# In[28]:


from sklearn.linear_model import LinearRegression

train_df = df[df['global_radiation'].notna() & df['sunshine'].notna()]
X_train = train_df[['sunshine']]
y_train = train_df['global_radiation']

model = LinearRegression()
model.fit(X_train, y_train)

missing_df = df[df['global_radiation'].isna() & df['sunshine'].notna()]
X_missing = missing_df[['sunshine']]

predicted_radiation = model.predict(X_missing)
df.loc[missing_df.index, 'global_radiation'] = predicted_radiation


# In[32]:


null_counts = df.isnull().sum() #description of null values per column
print(null_counts)


# In[34]:


df[df['max_temp'].isna()]


# In[36]:


df['date'] = pd.to_datetime(df['date'], format='%Y%m%d', errors='coerce')
df['year'] = df['date'].dt.year
df['dayofyear'] = df['date'].dt.dayofyear


# In[37]:


missing_2020 = df[(df['year'] == 2020) & (df['max_temp'].isna())]

# Create a lookup table of average max_temp by dayofyear (excluding 2020)
historical_avg = df[(df['year'] != 2020) & (df['max_temp'].notna())].groupby('dayofyear')['max_temp'].mean()

# Map those averages back to the missing 2020 values
df.loc[missing_2020.index, 'max_temp'] = missing_2020['dayofyear'].map(historical_avg)


# In[38]:


null_counts = df.isnull().sum() #description of null values per column
print(null_counts)


# In[39]:





# In[41]:


# Define features and target
features = ['max_temp', 'min_temp', 'cloud_cover', 'sunshine', 'global_radiation']
target = 'mean_temp'

# Drop rows with missing values in features or target (for training)
df_train = df.dropna(subset=features + [target])

X = df_train[features]
y = df_train[target]


# In[44]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
print(f'Mean Squared Error on test set: {mse:.3f}')


# In[45]:


# Select rows where mean_temp is null but features are not null
df_missing = df[df[target].isna() & df[features].notna().all(axis=1)]

# Predict
predictions = model.predict(df_missing[features])

# Fill in the missing mean_temp values
df.loc[df_missing.index, target] = predictions


# In[46]:


print(f"Missing mean_temp values remaining: {df['mean_temp'].isna().sum()}")


# In[52]:


null_counts = df.isnull().sum() #description of null values per column
print(null_counts)


# In[49]:


# Assume df is DataFrame and 'date' is in a string format like 'DD/MM/YYYY'

# Step 1: Convert 'date' column to datetime
df['date'] = pd.to_datetime(df['date'], dayfirst=True)

# Step 2: Extract day and month to group by
df['day'] = df['date'].dt.day
df['month'] = df['date'].dt.month

# Step 3: Identify indices with missing mean_temp and min_temp
missing_indices = df[(df['mean_temp'].isna()) & (df['min_temp'].isna())].index

for idx in missing_indices:
    day = df.at[idx, 'day']
    month = df.at[idx, 'month']
    
    # Filter records with same day/month and non-null temps, excluding the missing record itself
    matching_records = df[
        (df['day'] == day) &
        (df['month'] == month) &
        (df.index != idx) &
        (df['mean_temp'].notna()) &
        (df['min_temp'].notna())
    ]
    
    # Calculate averages
    mean_temp_avg = matching_records['mean_temp'].mean()
    min_temp_avg = matching_records['min_temp'].mean()
    
    # Fill missing values
    df.at[idx, 'mean_temp'] = mean_temp_avg
    df.at[idx, 'min_temp'] = min_temp_avg

# Step 4: Drop helper columns
df.drop(columns=['day', 'month'], inplace=True)


# In[56]:


plt.figure(figsize=(10, 8))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Correlation Matrix')
plt.show()


# In[59]:


# Select features correlated with precipitation
features = ['cloud_cover', 'pressure', 'sunshine', 'global_radiation']

# Split data
train_df = df[df['precipitation'].notna()]
predict_df = df[df['precipitation'].isna()]

# Impute missing values in features (if any) with median
imputer = SimpleImputer(strategy='median')
X_train = imputer.fit_transform(train_df[features])
X_predict = imputer.transform(predict_df[features])

y_train = train_df['precipitation']

# Train model
model = RandomForestRegressor(random_state=42)
model.fit(X_train, y_train)

# Predict missing precipitation
predictions = model.predict(X_predict)

# Fill missing values in original df
df.loc[df['precipitation'].isna(), 'precipitation'] = predictions

print("Missing precipitation values filled.")


# In[61]:


null_counts = df.isnull().sum() #description of null values per column
print(null_counts)


# In[63]:


# Select features that may influence pressure
features = ['sunshine', 'global_radiation', 'cloud_cover',
            'max_temp', 'mean_temp', 'min_temp', 'precipitation']

# Separate the rows with and without missing pressure
df_known = df[df['pressure'].notnull()]
df_unknown = df[df['pressure'].isnull()]

X_train = df_known[features]
y_train = df_known['pressure']
X_pred = df_unknown[features]

# Fit the model
model = RandomForestRegressor(random_state=42)
model.fit(X_train, y_train)

# Predict missing pressure values
predicted_pressure = model.predict(X_pred)

# Fill in the missing values
df.loc[df['pressure'].isnull(), 'pressure'] = predicted_pressure


# In[9]:


null_counts = df.isnull().sum() #description of null values per column
print(null_counts)


# In[65]:


# Define the full path including file name and extension
file_path = r"C:\Users\Julie\Desktop\London Weather\London Weather EDA\Cleaned LW DS.csv"

# Save the DataFrame to a CSV file
df.to_csv(file_path, index=False)


# In[67]:


# Filter for rows with missing snow_depth
missing_snow_depth = df[df['snow_depth'].isnull()]

# Count missing values by year
missing_by_year = missing_snow_depth['year'].value_counts().sort_index()

# Plot
plt.figure(figsize=(12, 6))
missing_by_year.plot(kind='bar', color='skyblue', edgecolor='black')

plt.title('Number of Records with Missing snow_depth by Year')
plt.xlabel('Year')
plt.ylabel('Count of Missing Records')
plt.xticks(rotation=45)
plt.tight_layout()
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.show()


# In[69]:


# Ensure 'date' column is in datetime format and sorted
df['date'] = pd.to_datetime(df['date'], dayfirst=True)
df.sort_values('date', inplace=True)

# Plot snow depth over time
plt.figure(figsize=(16, 6))
plt.plot(df['date'], df['snow_depth'], color='royalblue', linewidth=1)

plt.title('Snow Depth Over Time')
plt.xlabel('Date')
plt.ylabel('Snow Depth (cm)')
plt.xticks(rotation=45)
plt.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()
plt.show()


# In[73]:


missing_percentage = df['snow_depth'].isna().mean() * 100
print(f"Missing snow_depth values: {missing_percentage:.2f}%")


# In[74]:


# First, ensure you have a 'year' column (e.g., extracted from a date column if needed)
# Assuming 'year' column already exists

# Group by year and check if any snow_depth value is missing in that year
complete_years = df.groupby('year')['snow_depth'].apply(lambda x: x.isna().sum() == 0)

# Filter to only years with complete data
complete_years = complete_years[complete_years].reset_index()

# Add count of such years
count_complete_years = complete_years.shape[0]

print(f"Number of years with complete snow_depth entries: {count_complete_years}")
print(complete_years)


# In[19]:


# Ensure 'date' column is in datetime format
df['date'] = pd.to_datetime(df['date'])

# Create a 'year' column
df['year'] = df['date'].dt.year

# Filter dataframe for years 1979 to 2004
df_filtered = df[(df['year'] >= 1979) & (df['year'] <= 2004)]

# Select only numeric columns for correlation
numeric_df = df_filtered.select_dtypes(include='number')

# Calculate correlation matrix
corr_matrix = numeric_df.corr()

# Plot heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap='coolwarm', cbar=True)
plt.title('Correlation Matrix (1979–2004)')
plt.show()


# In[90]:


# df.to_csv(r"C:\Users\Julie\Desktop\London Weather\London Weather EDA\CleanLWDS.csv", index=False)


# In[21]:


# Check for null values per column
print(df.isnull().sum())

# Check data types of each column
print(df.dtypes)


# In[95]:


# df.to_csv(r"C:\Users\Julie\Desktop\London Weather\London Weather EDA\CleanLWDS.csv", index=False)


# In[20]:


df.head()


# In[22]:


df['month_name'] = df['date'].dt.month_name()
df['weekday'] = df['date'].dt.day_name()


# In[23]:


df.head()


# In[26]:


# Ensure 'date' is datetime
df['date'] = pd.to_datetime(df['date'])

# Filter for data from 1979 onward
df = df[df['date'].dt.year >= 1979]

# Extract weekday names
df['weekday'] = df['date'].dt.day_name()

# Group by weekday and get the maximum temperature
weekday_max_temp = df.groupby('weekday')['max_temp'].max().reset_index()

# Sort from hottest to coldest
weekday_max_temp = weekday_max_temp.sort_values(by='max_temp', ascending=False)

# Set weekday as categorical to preserve order in the plot
weekday_max_temp['weekday'] = pd.Categorical(weekday_max_temp['weekday'],
                                             categories=weekday_max_temp['weekday'],
                                             ordered=True)

# Plot the ranked scatter plot
plt.figure(figsize=(10, 6))
sns.scatterplot(data=weekday_max_temp, x='weekday', y='max_temp', s=120, color='orange', edgecolor='black')

# Add labels and title
plt.title('Hottest Weekdays by Max Temperature (Since 1979)', fontsize=14)
plt.xlabel('Weekday (Ranked Hottest to Coldest)')
plt.ylabel('Max Temperature (°C)')
plt.grid(True)
plt.tight_layout()
plt.show()


# In[27]:


# Ensure 'date' is datetime
df['date'] = pd.to_datetime(df['date'])

# Filter for data from 1979 onward
df = df[df['date'].dt.year >= 1979]

# Extract weekday name
df['weekday'] = df['date'].dt.day_name()

# Group by weekday and calculate average precipitation
weekday_precip = df.groupby('weekday')['precipitation'].mean().reset_index()

# Sort from wettest to driest
weekday_precip = weekday_precip.sort_values(by='precipitation', ascending=False)

# Preserve the ranked order for plotting
weekday_precip['weekday'] = pd.Categorical(weekday_precip['weekday'],
                                           categories=weekday_precip['weekday'],
                                           ordered=True)

# Plot the scatter chart
plt.figure(figsize=(10, 6))
sns.scatterplot(data=weekday_precip, x='weekday', y='precipitation',
                s=120, color='skyblue', edgecolor='black')

plt.title('Average Precipitation by Weekday (Since 1979)', fontsize=14)
plt.xlabel('Weekday (Ranked Wettest to Driest)')
plt.ylabel('Average Daily Precipitation (mm)')
plt.grid(True)
plt.tight_layout()
plt.show()


# In[28]:


# Define function to assign seasons based on month
def get_season(month):
    if month in [12, 1, 2]:
        return 'Winter'
    elif month in [3, 4, 5]:
        return 'Spring'
    elif month in [6, 7, 8]:
        return 'Summer'
    else:
        return 'Autumn'

# Ensure date is datetime
df['date'] = pd.to_datetime(df['date'])

# Extract month and year
df['month'] = df['date'].dt.month
df['year'] = df['date'].dt.year

# Assign seasons
df['season'] = df['month'].apply(get_season)


# In[29]:


# Group by year and season, then calculate average values
seasonal_summary = df.groupby(['year', 'season'])[['max_temp', 'precipitation']].mean().reset_index()


# In[36]:


df.head()


# In[38]:


import seaborn as sns
import matplotlib.pyplot as plt

# Ensure 'season' and 'year' columns are present
# Group by year and season, then get the average temperature
seasonal_data = df.groupby(['year', 'season'])['mean_temp'].mean().unstack()

# Reorder columns (optional, to show Spring-Summer-Autumn-Winter)
season_order = ['Spring', 'Summer', 'Autumn', 'Winter']
seasonal_data = seasonal_data[season_order]

# Plot
plt.figure(figsize=(10, 6))
sns.heatmap(seasonal_data, annot=True, fmt=".1f", cmap='coolwarm', linewidths=0.5)

plt.title('Average Seasonal Temperature by Year')
plt.xlabel('Season')
plt.ylabel('Year')
plt.tight_layout()
plt.show()


# In[60]:


df.head()


# In[54]:


df.drop(columns=['snow_likelihood'], inplace=True)


# In[66]:


df.columns


# In[61]:


# df.to_csv(r"C:\Users\Julie\Desktop\London Weather\London Weather EDA\CleanLWDS.csv", index=False)


# In[68]:


plt.figure(figsize=(12, 6))
sns.barplot(
    data=top10_warmest,
    x='label',
    y='mean_temp',
    hue='label',           # Add hue
    palette='Oranges_r',
    dodge=False,
    legend=False           # Hide the legend since hue = x
)
plt.title('Top 10 Warmest Seasons (1979–2020)', fontsize=14)
plt.xlabel('Season and Year')
plt.ylabel('Average Temperature (°C)')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()


# In[72]:


import matplotlib.pyplot as plt
import seaborn as sns

# Step 1: Group data
season_summary = (
    df.groupby(['year', 'season'], as_index=False)
    .agg({
        'mean_temp': 'mean',
        'sunshine': 'mean'  # Optional: bubble size
    })
)

# Optional: order seasons
season_order = ['Winter', 'Spring', 'Summer', 'Autumn']
season_summary['season'] = pd.Categorical(season_summary['season'], categories=season_order, ordered=True)

# Step 2: Plot
plt.figure(figsize=(14, 6))
scatter = sns.scatterplot(
    data=season_summary,
    x='year',
    y='mean_temp',
    hue='season',
    size='sunshine',           # Optional: use sunshine as bubble size
    sizes=(40, 300),
    alpha=0.6,
    palette='Set2',
)

plt.title('Seasonal Average Temperatures (1979–2020)', fontsize=14)
plt.xlabel('Year')
plt.ylabel('Average Temperature (°C)')
plt.legend(title='Season', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.show()


# In[75]:


import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Step 1: Create the average temperature feature
df['avg_temp_all'] = df[['max_temp', 'mean_temp', 'min_temp']].mean(axis=1)

# Step 2: Group by year and sort by ascending year
yearly_avg = df.groupby('year')[['avg_temp_all', 'global_radiation']].mean().reset_index()
yearly_avg = yearly_avg.sort_values(by='year')

# Step 3: Calculate correlation
correlation = yearly_avg[['avg_temp_all', 'global_radiation']].corr().iloc[0, 1]
print(f"Pearson correlation (annual avg): {correlation:.3f}")

# Step 4: Plot with line and colour gradient
plt.figure(figsize=(12, 6))
points = plt.scatter(
    yearly_avg['avg_temp_all'],
    yearly_avg['global_radiation'],
    c=yearly_avg['year'],
    cmap='viridis',
    s=100,
    edgecolor='black'
)

# Draw a line connecting the years in order
plt.plot(yearly_avg['avg_temp_all'], yearly_avg['global_radiation'], color='grey', linestyle='--', alpha=0.5)

# Annotate each point with the year (optional)
for _, row in yearly_avg.iterrows():
    plt.text(row['avg_temp_all'] + 0.02, row['global_radiation'] + 0.02, str(int(row['year'])), fontsize=7)

plt.colorbar(points, label='Year')
plt.title(f'Yearly Avg Global Radiation vs Avg Temperature\n(Pearson r = {correlation:.2f})')
plt.xlabel('Yearly Average Temperature (°C)')
plt.ylabel('Yearly Average Global Radiation (W/m²)')
plt.grid(True)
plt.tight_layout()
plt.show()


# In[76]:


df.head()


# In[78]:


import pandas as pd
import matplotlib.pyplot as plt

# Sum precipitation for each year-month combo
year_month_precip = df.groupby(['year', 'month_name'])['precipitation'].sum().reset_index()

# To keep months in calendar order
month_order = ['January', 'February', 'March', 'April', 'May', 'June', 
               'July', 'August', 'September', 'October', 'November', 'December']
year_month_precip['month_name'] = pd.Categorical(year_month_precip['month_name'], categories=month_order, ordered=True)

# Create a combined label for plotting (e.g. "Jan 1980")
year_month_precip['label'] = year_month_precip['month_name'].astype(str) + ' ' + year_month_precip['year'].astype(str)

# Top 10 wettest months
top10_wettest = year_month_precip.sort_values('precipitation', ascending=False).head(10)

# Top 10 driest months
top10_driest = year_month_precip.sort_values('precipitation', ascending=True).head(10)

# Plotting
plt.figure(figsize=(14,6))

plt.subplot(1, 2, 1)
plt.barh(top10_wettest['label'], top10_wettest['precipitation'], color='royalblue')
plt.title('Top 10 Wettest Months (Year-Month)')
plt.xlabel('Total Precipitation')
plt.gca().invert_yaxis()  # Highest on top

plt.subplot(1, 2, 2)
plt.barh(top10_driest['label'], top10_driest['precipitation'], color='orange')
plt.title('Top 10 Driest Months (Year-Month)')
plt.xlabel('Total Precipitation')
plt.gca().invert_yaxis()

plt.tight_layout()
plt.show()


# In[79]:


import pandas as pd
import matplotlib.pyplot as plt

# Normalize cloud_cover and sunshine
df['cloud_cover_norm'] = (df['cloud_cover'] - df['cloud_cover'].min()) / (df['cloud_cover'].max() - df['cloud_cover'].min())
df['sunshine_norm'] = (df['sunshine'] - df['sunshine'].min()) / (df['sunshine'].max() - df['sunshine'].min())

# Create gloominess feature
df['gloominess'] = df['cloud_cover_norm'] - df['sunshine_norm']

# Aggregate by year and month
gloomy_months = df.groupby(['year', 'month_name'])['gloominess'].mean().reset_index()

# Order months properly
month_order = ['January', 'February', 'March', 'April', 'May', 'June', 
               'July', 'August', 'September', 'October', 'November', 'December']
gloomy_months['month_name'] = pd.Categorical(gloomy_months['month_name'], categories=month_order, ordered=True)

# Create label for plotting
gloomy_months['label'] = gloomy_months['month_name'].astype(str) + ' ' + gloomy_months['year'].astype(str)

# Sort by gloominess descending and take top 10
top10_gloomy = gloomy_months.sort_values('gloominess', ascending=False).head(10)

# Plot
plt.figure(figsize=(10,6))
plt.barh(top10_gloomy['label'], top10_gloomy['gloominess'], color='slategray')
plt.xlabel('Average Gloominess (Normalized Cloud Cover - Sunshine)')
plt.title('Top 10 Gloomiest Months (Year-Month)')
plt.gca().invert_yaxis()  # Highest gloominess at top
plt.tight_layout()
plt.show()


# In[80]:


import pandas as pd
import matplotlib.pyplot as plt

# Normalize global_radiation, sunshine, and cloud_cover
df['radiation_norm'] = (df['global_radiation'] - df['global_radiation'].min()) / (df['global_radiation'].max() - df['global_radiation'].min())
df['sunshine_norm'] = (df['sunshine'] - df['sunshine'].min()) / (df['sunshine'].max() - df['sunshine'].min())
df['cloud_cover_norm'] = (df['cloud_cover'] - df['cloud_cover'].min()) / (df['cloud_cover'].max() - df['cloud_cover'].min())

# Create brightness feature as average of radiation, sunshine, and inverse cloud cover
df['brightness'] = (df['radiation_norm'] + df['sunshine_norm'] + (1 - df['cloud_cover_norm'])) / 3

# Aggregate by year and month
bright_months = df.groupby(['year', 'month_name'])['brightness'].mean().reset_index()

# Order months properly
month_order = ['January', 'February', 'March', 'April', 'May', 'June', 
               'July', 'August', 'September', 'October', 'November', 'December']
bright_months['month_name'] = pd.Categorical(bright_months['month_name'], categories=month_order, ordered=True)

# Create label for plotting
bright_months['label'] = bright_months['month_name'].astype(str) + ' ' + bright_months['year'].astype(str)

# Sort by brightness descending and take top 10
top10_bright = bright_months.sort_values('brightness', ascending=False).head(10)

# Plot
plt.figure(figsize=(10,6))
plt.barh(top10_bright['label'], top10_bright['brightness'], color='goldenrod')
plt.xlabel('Average Brightness (Radiation + Sunshine + Low Cloud Cover)')
plt.title('Top 10 Brightest Months (Year-Month)')
plt.gca().invert_yaxis()  # Brightest on top
plt.tight_layout()
plt.show()


# In[81]:


import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Assume df already has 'year', 'season', 'max_temp', 'mean_temp', 'min_temp'

# Create 5-year bins from 1979 to 2020
bins = list(range(1979, 2025, 5))  # e.g., 1979, 1984, 1989, ..., 2024
labels = [f'{b}-{b+4}' for b in bins[:-1]]

df['year_bin'] = pd.cut(df['year'], bins=bins, labels=labels, right=False)

# Group by year_bin and season, calculate mean temps
seasonal_temps = df.groupby(['year_bin', 'season'])[['max_temp', 'mean_temp', 'min_temp']].mean().reset_index()

# Plot seasonal trends for mean_temp as example
plt.figure(figsize=(12, 7))
seasons = seasonal_temps['season'].unique()
colors = ['red', 'green', 'blue', 'orange']

for i, season in enumerate(seasons):
    data = seasonal_temps[seasonal_temps['season'] == season]
    plt.plot(data['year_bin'], data['mean_temp'], label=season, color=colors[i])

plt.xlabel('5-Year Period')
plt.ylabel('Average Mean Temperature (°C)')
plt.title('Seasonal Mean Temperature Trends (5-Year Smoothed)')
plt.xticks(rotation=45)
plt.legend(title='Season')
plt.tight_layout()
plt.show()


# In[82]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Sample: assume df has 'season' and features: 'max_temp', 'mean_temp', 'min_temp', 'precipitation', 'cloud_cover'

# 1. Calculate seasonal averages
seasonal_avg = df.groupby('season')[['max_temp', 'mean_temp', 'min_temp', 'precipitation', 'cloud_cover']].mean()

# 2. Prepare radar plot parameters
categories = list(seasonal_avg.columns)
N = len(categories)

# Compute angle for each axis in the plot (divide the plot / number of variables)
angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
angles += angles[:1]  # complete the loop

# 3. Initialise plot
fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))

# 4. Plot each season
for season in seasonal_avg.index:
    values = seasonal_avg.loc[season].tolist()
    values += values[:1]  # complete the loop
    ax.plot(angles, values, label=season)
    ax.fill(angles, values, alpha=0.25)

# 5. Add category labels
ax.set_xticks(angles[:-1])
ax.set_xticklabels(categories)

# 6. Add title and legend
plt.title('Average Seasonal Weather Features')
plt.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))

plt.show()


# In[85]:


import numpy as np
import matplotlib.pyplot as plt

# Filter for Summer season only
summer_df = df[df['season'] == 'Summer']

# Group by year and calculate mean temperature
summer_yearly_avg_temp = summer_df.groupby('year')['mean_temp'].mean()

# Prepare data for linear regression
x = summer_yearly_avg_temp.index.values  # years as integers
y = summer_yearly_avg_temp.values

# Fit linear trend line (degree=1 polynomial)
coefficients = np.polyfit(x, y, 1)
trendline = np.poly1d(coefficients)

# Plot
plt.figure(figsize=(10,5))
plt.plot(x, y, marker='o', linestyle='-', label='Avg Summer Temp')
plt.plot(x, trendline(x), color='red', linestyle='--', label='Trendline')

plt.title('Average Summer Temperature Change with Trendline (1979-2020)')
plt.xlabel('Year')
plt.ylabel('Average Temperature (°C)')
plt.legend()
plt.grid(True)
plt.show()


# In[87]:


df.head()


# In[88]:


df.drop(columns=['year_bin'], inplace=True)


# In[91]:


df.columns


# In[92]:


df.to_csv(r"C:\Users\Julie\Desktop\London Weather\London Weather EDA\CleanLWDS.csv", index=False)


# In[98]:


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Parse date with UK-style format
df['date'] = pd.to_datetime(df['date'], dayfirst=True, errors='coerce')

# Filter for December 2010
dec_2010 = df[(df['date'].dt.year == 2010) & (df['date'].dt.month == 12)]

# Check snow_depth availability
print("Snow depth missing values in Dec 2010:", dec_2010['snow_depth'].isna().sum())
print("Total days in Dec 2010:", len(dec_2010))

# Plot
plt.figure(figsize=(10, 4))
sns.lineplot(data=dec_2010, x='date', y='snow_depth', marker='o', linewidth=2)
plt.title('Snow Depth in London – December 2010')
plt.xlabel('Date')
plt.ylabel('Snow Depth (cm)')
plt.xticks(rotation=45)
plt.tight_layout()
plt.grid(True)
plt.show()



# In[96]:


df.head()


# In[99]:


# Drop rows with missing snow_depth
snow_data = df.dropna(subset=['snow_depth'])

# Sort by snow_depth descending
top_snow = snow_data.sort_values(by='snow_depth', ascending=False)

# Get the top 10
top_10_snow = top_snow[['date', 'snow_depth']].head(10)

# Display the result
print("Top 10 highest snow depths:")
print(top_10_snow)


# In[103]:


# Drop rows with missing snow_depth or min_temp
snow_data = df.dropna(subset=['snow_depth', 'min_temp'])

# Sort by snow depth descending
top_snow = snow_data.sort_values(by='snow_depth', ascending=False)

# Select top 10 with relevant columns
top_10_snow = top_snow[['date', 'snow_depth', 'min_temp']].head(10)

import matplotlib.pyplot as plt

# Scatter plot of min temp vs snow depth
plt.figure(figsize=(8, 5))
plt.scatter(top_10_snow['min_temp'], top_10_snow['snow_depth'], color='dodgerblue', s=80)

# Add date labels
for i, row in top_10_snow.iterrows():
    plt.text(row['min_temp'], row['snow_depth'] + 0.5, row['date'].strftime('%Y-%m-%d'), fontsize=8, ha='center')

plt.title("Top 10 Snow Depth Days vs. Minimum Temperature")
plt.xlabel("Minimum Temperature (°C)")
plt.ylabel("Snow Depth (cm)")
plt.grid(True)
plt.tight_layout()
plt.show()


# In[102]:


df.columns


# In[104]:


# Drop rows with missing precipitation or min_temp
precip_data = df.dropna(subset=['precipitation', 'min_temp'])

# Sort by precipitation descending
top_precip = precip_data.sort_values(by='precipitation', ascending=False)

# Select top 10 with relevant columns
top_10_precip = top_precip[['date', 'precipitation', 'min_temp']].head(10)

import matplotlib.pyplot as plt

plt.figure(figsize=(8, 5))
plt.scatter(top_10_precip['min_temp'], top_10_precip['precipitation'], color='seagreen', s=80)

# Add date labels above points
for i, row in top_10_precip.iterrows():
    plt.text(row['min_temp'], row['precipitation'] + 0.1, row['date'].strftime('%Y-%m-%d'), fontsize=8, ha='center')

plt.title("Top 10 Precipitation Days vs. Minimum Temperature")
plt.xlabel("Minimum Temperature (°C)")
plt.ylabel("Precipitation (mm)")
plt.grid(True)
plt.tight_layout()
plt.show()


# In[105]:


# Ensure 'date' is datetime and filter the years 1979-2020
df['date'] = pd.to_datetime(df['date'], dayfirst=True, errors='coerce')
df_filtered = df[(df['date'].dt.year >= 1979) & (df['date'].dt.year <= 2020)]

# Drop rows with missing precipitation
precip_filtered = df_filtered.dropna(subset=['precipitation'])

# Find min and max precipitation and their dates
min_precip_row = precip_filtered.loc[precip_filtered['precipitation'].idxmin()]
max_precip_row = precip_filtered.loc[precip_filtered['precipitation'].idxmax()]

print("Lowest precipitation:")
print(f"Date: {min_precip_row['date'].date()}, Precipitation: {min_precip_row['precipitation']} mm")

print("\nHighest precipitation:")
print(f"Date: {max_precip_row['date'].date()}, Precipitation: {max_precip_row['precipitation']} mm")


# In[107]:


# Step 1: Define rain intensity categories
def rain_intensity(precip):
    if precip == 0:
        return 'No Rain'
    elif 0 < precip <= 2.5:
        return 'Light Rain'
    elif 2.5 < precip <= 10:
        return 'Moderate Rain'
    elif 10 < precip <= 30:
        return 'Heavy Rain'
    else:
        return 'Very Heavy Rain'

# Apply to dataset
df['rain_intensity'] = df['precipitation'].apply(rain_intensity)

# Step 2: Get top 10 rainiest days (drop missing precipitation)
top_rain = df.dropna(subset=['precipitation']).sort_values(by='precipitation', ascending=False).head(10)

# Step 3: Display the results with relevant columns
top_rain_days = top_rain[['date', 'precipitation', 'rain_intensity']]

print("Top 10 Rainiest Days:")
print(top_rain_days.to_string(index=False))


# In[116]:


heathrow_area_m2 = 12_270_000  # Heathrow area in square metres

# Calculate litres of rainwater per day at Heathrow
df['rain_litres'] = df['precipitation'] * heathrow_area_m2

# Filter date range
df_period = df[(df['date'].dt.year >= 1979) & (df['date'].dt.year <= 2020)]

# Sum total litres over period
total_litres = df_period['rain_litres'].sum()

print(f"Total rainwater over 1979-2020 at Heathrow: {total_litres:,.0f} litres")


# In[117]:


top_10_max_temp = df[['date', 'max_temp']].dropna().sort_values(by='max_temp', ascending=False).head(10)
print(top_10_max_temp)


# In[118]:


def temp_category(t):
    if t < 15:
        return 'Cool'
    elif 15 <= t < 20:
        return 'Lightly Warm'
    elif 20 <= t < 25:
        return 'Warm'
    elif 25 <= t < 30:
        return 'Very Warm'
    elif 30 <= t < 35:
        return 'Hot'
    else:
        return 'Very Hot'

df['temp_category'] = df['max_temp'].apply(temp_category)


# In[119]:


hot_days = df[df['max_temp'] > 30][['date', 'max_temp']]
print(hot_days)


# In[120]:


# First, create a decade column (e.g., 1980, 1990, 2000)
df['decade'] = (df['date'].dt.year // 10) * 10

# Calculate average max temperature per decade
avg_max_temp_decade = df.groupby('decade')['max_temp'].mean().reset_index()

# Sort by descending average max temp
avg_max_temp_decade = avg_max_temp_decade.sort_values(by='max_temp', ascending=False)

print(avg_max_temp_decade)


# In[121]:


# Make sure 'date' is datetime type
df['date'] = pd.to_datetime(df['date'])

# Extract year
df['year'] = df['date'].dt.year

# Calculate annual average max temperature
annual_max_temp = df.groupby('year')['max_temp'].mean().reset_index()

# Calculate percentage change from previous year
annual_max_temp['pct_change_max_temp'] = annual_max_temp['max_temp'].pct_change() * 100

print(annual_max_temp.head(10))


# In[122]:


import matplotlib.pyplot as plt
import seaborn as sns

# Ensure 'date' is datetime
df['date'] = pd.to_datetime(df['date'])

# Group by year and calculate mean max temperature
annual_avg = df.groupby(df['date'].dt.year)['max_temp'].mean().reset_index()
annual_avg.columns = ['year', 'avg_max_temp']

# Calculate percentage change year on year
annual_avg['pct_change'] = annual_avg['avg_max_temp'].pct_change() * 100

# Plotting
plt.figure(figsize=(14, 6))
sns.lineplot(data=annual_avg, x='year', y='avg_max_temp', marker='o', label='Avg Max Temp (°C)')

# Optional: annotate swings
for i in range(1, len(annual_avg)):
    plt.annotate(f"{annual_avg['pct_change'].iloc[i]:.1f}%",
                 (annual_avg['year'].iloc[i], annual_avg['avg_max_temp'].iloc[i]),
                 textcoords="offset points", xytext=(0,10), ha='center', fontsize=8, color='gray')

plt.title('Year-on-Year Average Max Temperature at Heathrow')
plt.xlabel('Year')
plt.ylabel('Avg Max Temperature (°C)')
plt.grid(True)
plt.tight_layout()
plt.show()


# In[123]:


import matplotlib.pyplot as plt
import seaborn as sns

# Prepare data
df['date'] = pd.to_datetime(df['date'])
annual_avg = df.groupby(df['date'].dt.year)['max_temp'].mean().reset_index()
annual_avg.columns = ['year', 'avg_max_temp']
annual_avg['pct_change'] = annual_avg['avg_max_temp'].pct_change() * 100

# Drop first row with NaN change
annual_avg = annual_avg.dropna()

# Plot
plt.figure(figsize=(14, 7))
colors = annual_avg['pct_change'].apply(lambda x: 'red' if x < 0 else 'green')

bars = plt.bar(annual_avg['year'], annual_avg['pct_change'], color=colors)

# Add zero line
plt.axhline(0, color='black', linewidth=0.8)

# Labels
plt.title('Year-on-Year % Change in Avg Max Temperature at Heathrow (1979–2020)', fontsize=14)
plt.xlabel('Year')
plt.ylabel('% Change from Previous Year', fontsize=12)
plt.grid(axis='y', linestyle='--', alpha=0.5)
plt.tight_layout()
plt.show()


# In[124]:


# Assuming annual_avg is already created with 'pct_change' column
annual_avg['abs_change'] = annual_avg['pct_change'].abs()

# Find the row with the biggest absolute change
biggest_swing = annual_avg.loc[annual_avg['abs_change'].idxmax()]

print("Year with biggest swing:")
print(f"Year: {int(biggest_swing['year'])}")
print(f"Change: {biggest_swing['pct_change']:.2f}%")
print(f"From: {annual_avg.loc[annual_avg.index == biggest_swing.name - 1, 'avg_max_temp'].values[0]:.2f}°C "
      f"to {biggest_swing['avg_max_temp']:.2f}°C")


# In[125]:


import pandas as pd
import matplotlib.pyplot as plt

# Ensure 'date' column is datetime
df['date'] = pd.to_datetime(df['date'])

# Calculate annual average mean temperature
annual_mean = df.groupby(df['date'].dt.year)['mean_temp'].mean().reset_index()
annual_mean.columns = ['year', 'avg_temp']

# Calculate year-on-year percentage change
annual_mean['pct_change'] = annual_mean['avg_temp'].pct_change() * 100

# Drop the first row with NaN pct_change
annual_mean = annual_mean.dropna()

# Find biggest absolute swing
annual_mean['abs_change'] = annual_mean['pct_change'].abs()
biggest_swing = annual_mean.loc[annual_mean['abs_change'].idxmax()]

print(f"📈 Largest swing was in {int(biggest_swing['year'])}: {biggest_swing['pct_change']:.2f}% change")

# Plot diverging bar chart
colors = annual_mean['pct_change'].apply(lambda x: 'green' if x > 0 else 'red')

plt.figure(figsize=(14, 6))
plt.bar(annual_mean['year'], annual_mean['pct_change'], color=colors)
plt.axhline(0, color='black', linewidth=0.8)
plt.title('Year-on-Year % Change in Average Temperature (London Heathrow)', fontsize=14)
plt.xlabel('Year')
plt.ylabel('% Change from Previous Year', fontsize=12)
plt.grid(axis='y', linestyle='--', alpha=0.5)
plt.tight_layout()
plt.show()


# In[126]:


top5_swings = annual_mean.sort_values(by='abs_change', ascending=False).head(5)

print("Top 5 biggest year-on-year temperature changes:")
print(top5_swings[['year', 'pct_change']])


# In[127]:


# Convert date column to datetime if not already
df['date'] = pd.to_datetime(df['date'])

# Top 5 hottest days (highest max_temp)
top5_hottest = df.nlargest(5, 'max_temp')[['date', 'max_temp']]

# Top 5 coldest days (lowest min_temp)
top5_coldest = df.nsmallest(5, 'min_temp')[['date', 'min_temp']]

print("Top 5 Hottest Days:")
print(top5_hottest)

print("\nTop 5 Coldest Days:")
print(top5_coldest)


# In[128]:


# Ensure 'date' column is datetime
df['date'] = pd.to_datetime(df['date'])

# Filter data for 2011
df_2011 = df[df['date'].dt.year == 2011]

# Find the hottest day in 2011
hottest_day_2011 = df_2011.loc[df_2011['max_temp'].idxmax(), ['date', 'max_temp']]

print("Hottest day in 2011:")
print(hottest_day_2011)


# In[129]:


df.columns


# In[130]:


save_path = r"C:\Users\Julie\Desktop\London Weather\London Weather EDA\Day 3\Cleaned_LW.csv"
df.to_csv(save_path, index=False)


# In[131]:


# Calculate the daily temperature range, aka Diurnal Temperature Range
df['dtr'] = df['max_temp'] - df['min_temp']


# In[132]:


df.head()


# In[134]:


avg_dtr_by_season = df.groupby('season')['dtr'].mean().sort_values()

avg_dtr_by_season.plot(kind='bar', color='coral', title='Average DTR by Season')
plt.ylabel('DTR (°C)')
plt.grid(axis='y')
plt.tight_layout()
plt.show()


# In[138]:


top5_dtr = df.nlargest(5, 'dtr')[['date', 'max_temp', 'min_temp', 'dtr', 'cloud_cover']]
print("Top 5 Highest DTR Days:\n", top5_dtr)


# In[141]:


# Filter for September days
sept_days = df[df['month'] == 9]

# Get top 5 hottest days in September by max_temp
top5_hottest_sept_days = sept_days.nlargest(5, 'max_temp')[['date', 'max_temp', 'min_temp', 'mean_temp']]

print("Top 5 Hottest September Days:")
print(top5_hottest_sept_days)



# In[143]:


df.head()


# In[ ]:


# Create a decade column, e.g., 1980s, 1990s, etc.
df['decade'] = (df['year'] // 10) * 10

# Sum total precipitation per decade
total_precip_by_decade = df.groupby('decade')['precipitation'].sum().sort_values(ascending=False)

print("Total Precipitation by Decade:")
print(total_precip_by_decade)

# Optional: Count days by rain_intensity per decade
rain_counts_by_decade = df.groupby(['decade', 'rain_intensity']).size().unstack(fill_value=0)

print("\nRain Intensity Counts by Decade:")
print(rain_counts_by_decade)

# To see which decades had most heavy rain:
heavy_rain_by_decade = rain_counts_by_decade.get('Very Heavy Rain', None)
if heavy_rain_by_decade is not None:
    print("\nVery Heavy Rain days by Decade:")
    print(heavy_rain_by_decade.sort_values(ascending=False))

