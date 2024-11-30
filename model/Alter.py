import pandas as pd

# Load the dataset 
df = pd.read_csv('Dataset.csv')

# Define the base prompt that will be added before each input
prompt_template = (
    "Generate business report contents for the company 'ABC Corp.' based on the following scope: "
    "{scope} \n\n"
    "Generate the following sections:\n"
    "- Executive Summary\n"
    "- Industry Overview and Trends\n"
    "- Problem Statement\n"
    "- Proposed Solution\n"
    "- Market Analysis\n"
    "- Sustainable Practices\n"
    "- Supply Chain and Distribution\n"
    "- Financial Projections\n"
    "- Implementation Timeline\n"
    "- Conclusion\n"
)

# Apply the transformation to the "User input English-in English letters" column
def format_prompt(row):
    return prompt_template.format(scope=row['User input English-in English letters'])

# Apply the transformation to the whole column
df['User input English-in English letters'] = df.apply(format_prompt, axis=1)

# Select the desired columns for the new dataset
df_transformed = df[['Business Name', 'Domain', 'User input English-in English letters', 'System output']]

# Save the transformed dataset to a new Excel file
df_transformed.to_excel('transformed_dataset.xlsx', index=False)

print("Dataset transformation complete! Saved to 'transformed_dataset.xlsx'")
