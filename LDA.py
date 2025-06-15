
# Libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sb
from scipy.stats import shapiro
from scipy import stats
from sklearn.preprocessing import StandardScaler
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
import io
import base64
from dash import dcc
import plotly.graph_objs as go
import numpy as np

# Loading data
data_agri = pd.read_csv("Dane_lic(nb_b_agriculture_16_year_13).csv", sep=";", decimal=".")
#print(data_agri)

# Basic descriptive statistics

stats1 = data_agri.describe()
#print(stats1)

# Histograms

data_agri_numerals = data_agri.select_dtypes(exclude=['object'])

def all_histograms_base64(data):
    fig = data.hist(figsize=(18, 12), bins=10)
    buf = io.BytesIO()
    plt.tight_layout()
    plt.savefig(buf, format='png')
    buf.seek(0)
    encoded = base64.b64encode(buf.read()).decode()
    plt.close()  # very important: avoid memory leak
    return f'data:image/png;base64,{encoded}'
histograms_image = all_histograms_base64(data_agri_numerals)

# Boxplots
def boxplots_base64(data):
    fig, axs = plt.subplots(nrows=5, ncols=5, figsize=(20, 15))
    axs = axs.flatten()
    for i, column in enumerate(data.columns):
        sb.boxplot(y=data[column], ax=axs[i])
        axs[i].set_title(column)
    for j in range(i + 1, len(axs)):
        fig.delaxes(axs[j])
    buf = io.BytesIO()
    plt.tight_layout()
    plt.savefig(buf, format='png')
    buf.seek(0)
    encoded = base64.b64encode(buf.read()).decode()
    plt.close()
    return f'data:image/png;base64,{encoded}'




# Shapiro-Wilk test for each variable to check if data follows a normal distribution
# Create an empty list to collect the results
shapiro_results = []

# Loop through each numeric column
for column in data_agri_numerals.select_dtypes(include=['float64', 'int64']).columns:
    stat, p_value = shapiro(data_agri_numerals[column])
    normality = "Yes" if p_value > 0.05 else "No"
    shapiro_results.append({
        "Variable": column,
        "Shapiro-Wilk p-value": round(p_value, 4),
        "Normally Distributed?": normality
    })

# Convert to DataFrame
shapiro_df = pd.DataFrame(shapiro_results)
shapiro_df = shapiro_df.reset_index(drop=True)

# Check if the data contains non-positive values
if (data_agri_numerals <= 0).any().any():
    print("Data contains non-positive values. Shifting the data to make all values positive.")

    # Shift the data by adding the absolute value of the minimum value + 1 to make everything positive
    for column in data_agri_numerals.select_dtypes(include=['float64', 'int64']).columns:
        min_value = data_agri_numerals[column].min()

        # Shift values if minimum value is <= 0
        if min_value <= 0:
            shift_value = abs(min_value) + 1
            data_agri_numerals[column] += shift_value  # Shift values to positive

# Apply Box-Cox transformation to each column
for column in data_agri_numerals.select_dtypes(include=['float64', 'int64']).columns:
    # Perform Box-Cox transformation (this will return the transformed data and lambda)
    transformed_data, _ = stats.boxcox(data_agri_numerals[column])
    data_agri_numerals[column] = transformed_data  # Update the column with transformed data

# Check skewness after Box-Cox transformation
skew_values_after_boxcox = data_agri_numerals.skew()
skew_df = pd.DataFrame(skew_values_after_boxcox, columns=['Skewness']).reset_index()
skew_df.columns = ['Variable', 'Skewness']


# Function to perform the Shapiro-Wilk test for normality
def check_normality(data):
    p_values = []
    for column in data.columns:
        _, p_value = shapiro(data[column].dropna())  # Drop NaN values before testing
        p_values.append(p_value)
    return pd.Series(p_values, index=data.columns)

# Test normality of all variables in data_agri_numerals
p_values = check_normality(data_agri_numerals)

# List columns that do not follow normal distribution (p-value < 0.05)
columns_to_remove = p_values[p_values < 0.05].index

# Remove non-normal columns from data_agri_numerals
data_agri_numerals_cleaned = data_agri_numerals.drop(columns=columns_to_remove)

# Print cleaned data
#print(data_agri_numerals_cleaned)

##Boxplots
boxplot_image = boxplots_base64(data_agri_numerals_cleaned)


# Removing the variable (column) V17 from the DataFrame
data_agri_numerals_cleaned = data_agri_numerals_cleaned.drop(columns=['V17'])

# plt.figure(figsize=(20, 15))
#
# for i, column in enumerate(data_agri_numerals_cleaned.columns, 1):
#     plt.subplot(5, 5, i)
#     sb.boxplot(y=data_agri_numerals_cleaned[column])
# plt.tight_layout()
# #plt.show()


# Remove the 10th row (index 9) from the cleaned DataFrame
# This row contains mostly zeros and causes outlier behavior in the analysis,
# especially when preparing data for LDA. Removing it improves data quality.
data_agri_numerals_cleaned = data_agri_numerals_cleaned.drop(index=9)

# Optional: reset the index after dropping the row
data_agri_numerals_cleaned = data_agri_numerals_cleaned.reset_index(drop=True)

plt.figure(figsize=(20, 15))

for i, column in enumerate(data_agri_numerals_cleaned.columns, 1):
    plt.subplot(5, 5, i)
    sb.boxplot(y=data_agri_numerals_cleaned[column])
plt.tight_layout()
#plt.show()


# Function to replace outliers with the median using the IQR method
def replace_outliers_with_median(df):
    df_cleaned = df.copy()
    for column in df.columns:
        Q1 = df[column].quantile(0.25)
        Q3 = df[column].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR

        # Find outliers
        outliers = (df[column] < lower_bound) | (df[column] > upper_bound)

        # Replace outliers with median
        median = df[column].median()
        df_cleaned.loc[outliers, column] = median

        #print(f"{column}: Replaced {outliers.sum()} outlier(s) with median.")

    return df_cleaned

# Apply the function
data_agri_numerals_cleaned = replace_outliers_with_median(data_agri_numerals_cleaned)
data_agri_numerals_cleaned = replace_outliers_with_median(data_agri_numerals_cleaned)

# Removing the variable (column) V1 from the DataFrame because it contains too many outliers,
# which negatively impact the analysis
data_agri_numerals_cleaned = data_agri_numerals_cleaned.drop(columns=['V1'])

##Boxplots
boxplot_image2 = boxplots_base64(data_agri_numerals_cleaned)

# Calculate skewness for each variable in the dataset

skew_values = data_agri_numerals_cleaned.skew()
#print(skew_values)

# Shapiro–Wilk test for fully cleaned data
shapiro_final_results = []

for column in data_agri_numerals_cleaned.select_dtypes(include=['float64', 'int64']).columns:
    stat, p_value = shapiro(data_agri_numerals_cleaned[column])
    normality = "Yes" if p_value > 0.05 else "No"
    shapiro_final_results.append({
        "Variable": column,
        "Shapiro-Wilk p-value": round(p_value, 4),
        "Normally Distributed?": normality
    })

shapiro_final_df = pd.DataFrame(shapiro_final_results).reset_index(drop=True)
# Extract the 'Type' column which contains the bankruptcy status
bankrucy_status = data_agri['Type']
# Remove the same row(s) that were previously deleted from the main dataset (e.g. due to outliers)
bankrucy_status = bankrucy_status.drop(index=9)


# Reset the index of the bankruptcy_status Series to match the cleaned data
bankrucy_status = bankrucy_status.reset_index(drop=True)

# Reset the index of the cleaned numerical data to ensure alignment
data_agri_numerals_cleaned = data_agri_numerals_cleaned.reset_index(drop=True)

# Add the cleaned bankruptcy status column to the cleaned DataFrame
data_agri_numerals_cleaned['bankrucy_status'] = bankrucy_status

# Print the DataFrame to verify the new column has been added correctly
#print(data_agri_numerals_cleaned.head())


# Separate features (X) from the target variable (y)
X = data_agri_numerals_cleaned.drop(columns=['bankrucy_status'])
y = data_agri_numerals_cleaned['bankrucy_status']

# Initialize the scaler
scaler = StandardScaler()

# Fit the scaler on X and transform the data
X_scaled = scaler.fit_transform(X)

# Convert the result back to a DataFrame for easier use
X_scaled = pd.DataFrame(X_scaled, columns=X.columns)

# Print a few rows to check the result
#print(X_scaled.head())

# Perform LDA on the scaled data
lda = LinearDiscriminantAnalysis(n_components=1)  # Używamy n_components=1, bo są tylko 2 klasy
X_lda = lda.fit_transform(X_scaled, y)

# Get LDA coefficients
coefs = lda.coef_[0]
features = X.columns

# Assign colors: green for positive, red for negative coefficients
colors = ['green' if val > 0 else 'red' for val in coefs]

# Create bar chart
plt.figure(figsize=(12, 6))
bars = plt.bar(features, coefs, color=colors)

# Add a horizontal line at y=0 for reference
plt.axhline(0, color='black', linewidth=0.8)

# Add title and axis labels
plt.title('LDA Discriminant Coefficients')
plt.xlabel('Features')
plt.ylabel('Coefficient Value')

# Rotate x-axis labels for better readability
plt.xticks(rotation=45, ha='right')

# Adjust layout and save the figure to the assets folder for Dash
plt.tight_layout()
plt.savefig('assets/lda_coefficients.png', dpi=300)
plt.close()

# Convert the result to a DataFrame for plotting
lda_df = pd.DataFrame()
lda_df['LDA1'] = X_lda.ravel()  # ravel() to flatten array
lda_df['Status'] = y.values

# Plot the result
def lda_plot_base64(lda_df):
    plt.figure(figsize=(8, 6))
    sb.histplot(data=lda_df, x='LDA1', hue='Status', element='step', stat='density', common_norm=False)
    plt.title("LDA projection - separation between bankruptcy statuses")
    plt.xlabel("LDA Component 1")
    plt.ylabel("Density")
    plt.grid(True)

    buf = io.BytesIO()
    plt.tight_layout()
    plt.savefig(buf, format='png')
    buf.seek(0)
    encoded = base64.b64encode(buf.read()).decode()
    plt.close()
    return f'data:image/png;base64,{encoded}'


lda_img = lda_plot_base64(lda_df)
#plt.show()

coeffs = lda.coef_[0]
intercept = lda.intercept_[0]
features = X.columns  # or X_scaled.columns if it's a DataFrame

equation = " + ".join([f"{round(coef, 3)}·{name}" for coef, name in zip(coeffs, features)])
equation = f"LDA1 = {equation} + {round(intercept, 3)}"

lda_equation_latex = f"$$\\text{{LDA}}(x) = {equation}$$"

# === DASH GUI ===
import pandas as pd
import dash
from dash import dash_table, html, dcc

# Initialize the Dash app
app = dash.Dash(__name__)

# Define layout
app.layout = html.Div([
    html.H1(" Using Linear Discriminant Analysis to Assess the Financial Situation of Enterprises", style={'textAlign': 'center'}),

    html.H2("1. Basic Descriptive Statistics:  data.describe()", style={'marginTop': '20px'}),

    dash_table.DataTable(
        data=stats1.reset_index().to_dict('records'),
        columns=[{"name": i, "id": i} for i in stats1.reset_index().columns],
        style_table={'overflowX': 'auto'},
        style_cell={
            'textAlign': 'center',
            'padding': '5px',
            'minWidth': '100px'
        },
        style_header={
            'backgroundColor': '#f0f0f0',
            'fontWeight': 'bold'
        }
    ),

    html.H2("Histograms of All Variables", style={'textAlign': 'center'}),
    html.Img(src=histograms_image, style={'width': '90%', 'display': 'block', 'margin': 'auto'}),

    html.H3("Shapiro–Wilk Normality Test Results", style={'textAlign': 'center'}),
    dash_table.DataTable(
        # Reset index without adding the index as a column
        data=shapiro_df.reset_index(drop=True).to_dict('records'),

        # Columns now come from the reset dataframe without the index column
        columns=[{"name": col, "id": col} for col in shapiro_df.reset_index(drop=True).columns],
        style_table={'overflowX': 'auto'},
        style_cell={
            'textAlign': 'center',
            'padding': '5px',
            'minWidth': '100px'
        },
        style_header={
            'backgroundColor': '#f0f0f0',
            'fontWeight': 'bold'
        }
    ),
    html.P(
    "Based on the Shapiro–Wilk test results, most variables do not follow a normal distribution (p-value < 0.05). Therefore, to prepare the data for Box-Cox transformation—which requires strictly positive values—the variables were shifted to eliminate non-positive values. This step ensures that assumptions necessary for further analysis, such as Linear Discriminant Analysis (LDA), are met.",
    style={'textAlign': 'center', 'fontSize': '16px', 'marginTop': '20px'}
    ),
    html.H3("Skewness After Box-Cox Transformation", style={'textAlign': 'center'}),
    dash_table.DataTable(
    data=skew_df.to_dict('records'),
    columns=[{"name": i, "id": i} for i in skew_df.columns],
    style_table={'overflowX': 'auto'},
    style_cell={
        'textAlign': 'center',
        'padding': '5px',
        'minWidth': '100px'
    },
    style_header={
        'backgroundColor': '#f0f0f0',
        'fontWeight': 'bold'
    }

    ),

    html.H2("Data after Removing Non-Normal Variables and Cleaning", style={'textAlign': 'center'}),
    dash_table.DataTable(
    data=data_agri_numerals_cleaned.head(10).to_dict('records'),
    columns=[{"name": i, "id": i} for i in data_agri_numerals_cleaned.columns],
    style_table={'overflowX': 'auto'},
    style_cell={
        'textAlign': 'center',
        'padding': '5px',
        'minWidth': '100px'
    },
    style_header={
        'backgroundColor': '#f0f0f0',
        'fontWeight': 'bold'
    }
),
    html.H2("Boxplots After Cleaning", style={'textAlign': 'center', 'marginTop': '40px'}),
    html.Img(src=boxplot_image, style={'width': '90%', 'display': 'block', 'margin': 'auto'}),
    html.H2("Boxplot After Cleaning and Outlier Removal", style={'textAlign': 'center', 'marginTop': '40px'}),
    html.Img(src=boxplot_image2, style={'width': '90%', 'display': 'block', 'margin': 'auto'}),

html.H3("Final Shapiro–Wilk Test After Full Cleaning", style={'textAlign': 'center'}),
dash_table.DataTable(
    data=shapiro_final_df.to_dict('records'),
    columns=[{"name": i, "id": i} for i in shapiro_final_df.columns],
    style_table={'overflowX': 'auto'},
    style_cell={
        'textAlign': 'center',
        'padding': '5px',
        'minWidth': '100px'
    },
    style_header={
        'backgroundColor': '#f0f0f0',
        'fontWeight': 'bold'
    }
),
html.P(
    "After full cleaning (including removal of problematic variables and outliers), all variables now follow a normal distribution (p > 0.05). This ensures the dataset is fully ready for statistical modeling like LDA.",
    style={'textAlign': 'center', 'fontSize': '16px', 'marginTop': '20px'}
),
html.H2("LDA Projection: Bankruptcy Separation", style={'textAlign': 'center'}),
html.Img(src=lda_img, style={'width': '70%', 'display': 'block', 'margin': 'auto'}),
html.P(
    "The histogram shows a clear separation between the two classes (bankrupt and non-bankrupt) along the LDA axis. While there is a small overlapping region in the center, most of the distribution for each class is concentrated in distinct areas. This suggests that the LDA transformation provides good class discrimination, and the model may perform well in distinguishing between the two statuses.",
    style={'textAlign': 'center', 'fontSize': '16px', 'marginTop': '20px'}
    ),
html.H4("Linear Discriminant Function", style={'textAlign': 'center'}),
html.H2("LDA(x) = -9.049·V₅ + 17.07·V₆ − 8.17·V₇ + 0.2·V₈ + 7.814·V₉ − 4.344·V₁₀ + 2.555·V₁₄ + 2.942·V₁₅ + 0.952·V₁₆ + 6.102·V₂₀ − 6.37·V₂₁ + 0.866", style={'textAlign': 'center'}),

html.P("The equation below represents the first linear discriminant (LDA1), which is a linear combination of selected features. "
       "Each coefficient indicates the importance of a given variable in separating the two classes. "
       "Positive values increase the LDA1 score, while negative values decrease it. "
       "The result is used to distinguish between the two groups: a higher or lower score corresponds to one of the classes (NB or B)."),
html.H2("LDA Coefficients", style={'textAlign': 'center', 'marginTop': '40px'}),

html.Img(
    src='assets/lda_coefficients.png',
    style={'width': '80%', 'display': 'block', 'margin': 'auto'}
),

html.P(
    "The bar chart presents the coefficients of the linear discriminant function. "
    "Green bars indicate positive influence, red bars negative. "
    "The taller the bar, the stronger the feature's contribution to class separation.",
    style={'textAlign': 'center', 'fontSize': '16px', 'marginTop': '10px'}
),
html.P(
    "Variables with large positive coefficients contribute more to classifying a company as non-bankrupt, "
    "while variables with large negative coefficients increase the likelihood of a bankruptcy classification.",
    style={'textAlign': 'center', 'fontSize': '16px'}
),

html.P(
    "From the plot, it is clear that V₆ and V₂₀ have the highest positive impact, suggesting that high values in these features "
    "are associated with financially stable companies. In contrast, V₅ and V₇ have strong negative weights, indicating that they are key indicators of financial distress.",
    style={'textAlign': 'center', 'fontSize': '16px', 'marginBottom': '40px'}
),
])

# Run the server
if __name__ == '__main__':
    app.run(debug=True)
