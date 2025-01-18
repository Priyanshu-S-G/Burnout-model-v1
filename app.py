# import gradio as gr
# import pickle
# import numpy as np

# # Load the trained model
# with open("model.pkl", "rb") as file:
#     model = pickle.load(file)

# # Function to prepare input data and make predictions
# def predict_burnout(
#     age, daily_rate, distance_from_home, gender, hourly_rate, monthly_income,
#     monthly_rate, num_companies_worked, overtime, percent_salary_hike,
#     performance_rating, stock_option_level, total_working_years,
#     training_times_last_year, years_at_company, years_in_current_role,
#     years_since_last_promotion, years_with_curr_manager, business_travel,
#     department, education, education_field, environment_satisfaction,
#     job_involvement, job_level, job_role, job_satisfaction, marital_status,
#     relationship_satisfaction, work_life_balance
# ):
#     # Encode categorical variables
#     gender_binary = 1 if gender == "Male" else 0
#     overtime_binary = 1 if overtime == "Yes" else 0
#     business_travel_encoded = [
#         1 if business_travel == "Travel_Frequently" else 0,
#         1 if business_travel == "Travel_Rarely" else 0,
#     ]
#     department_encoded = [
#         1 if department == "Research & Development" else 0,
#         1 if department == "Sales" else 0,
#     ]
#     education_encoded = [
#         1 if education == "Below College" else 0,
#         1 if education == "College" else 0,
#         1 if education == "Doctor" else 0,
#         1 if education == "Master" else 0,
#     ]
#     education_field_encoded = [
#         1 if education_field == "Life Sciences" else 0,
#         1 if education_field == "Marketing" else 0,
#         1 if education_field == "Medical" else 0,
#         1 if education_field == "Other" else 0,
#         1 if education_field == "Technical Degree" else 0,
#     ]
#     environment_satisfaction_encoded = [
#         1 if environment_satisfaction == "Low" else 0,
#         1 if environment_satisfaction == "Medium" else 0,
#         1 if environment_satisfaction == "Very High" else 0,
#     ]
#     job_involvement_encoded = [
#         1 if job_involvement == "Low" else 0,
#         1 if job_involvement == "Medium" else 0,
#         1 if job_involvement == "Very High" else 0,
#     ]
#     job_level_encoded = [
#         1 if job_level == "Executive Level" else 0,
#         1 if job_level == "Junior Level" else 0,
#         1 if job_level == "Mid Level" else 0,
#         1 if job_level == "Senior Level" else 0,
#     ]
#     job_role_encoded = [
#         1 if job_role == "Human Resources" else 0,
#         1 if job_role == "Laboratory Technician" else 0,
#         1 if job_role == "Manager" else 0,
#         1 if job_role == "Manufacturing Director" else 0,
#         1 if job_role == "Research Director" else 0,
#         1 if job_role == "Research Scientist" else 0,
#         1 if job_role == "Sales Executive" else 0,
#         1 if job_role == "Sales Representative" else 0,
#     ]
#     marital_status_encoded = [
#         1 if marital_status == "Married" else 0,
#         1 if marital_status == "Single" else 0,
#     ]
#     relationship_satisfaction_encoded = [
#         1 if relationship_satisfaction == "Low" else 0,
#         1 if relationship_satisfaction == "Medium" else 0,
#         1 if relationship_satisfaction == "Very High" else 0,
#     ]
#     work_life_balance_encoded = [
#         1 if work_life_balance == "Best" else 0,
#         1 if work_life_balance == "Better" else 0,
#         1 if work_life_balance == "Good" else 0,
#     ]

#     # Combine all inputs
#     input_data = np.array([
#         age, daily_rate, distance_from_home, gender_binary, hourly_rate,
#         monthly_income, monthly_rate, num_companies_worked, overtime_binary,
#         percent_salary_hike, performance_rating, stock_option_level,
#         total_working_years, training_times_last_year, years_at_company,
#         years_in_current_role, years_since_last_promotion, years_with_curr_manager,
#         *business_travel_encoded, *department_encoded, *education_encoded,
#         *education_field_encoded, *environment_satisfaction_encoded,
#         *job_involvement_encoded, *job_level_encoded, *job_role_encoded,
#         *marital_status_encoded, *relationship_satisfaction_encoded,
#         *work_life_balance_encoded
#     ]).reshape(1, -1)

#     # Make predictions
#     prediction = model.predict(input_data)[0]
#     confidence = model.predict_proba(input_data)[0][prediction] * 100

#     return "Burnout" if prediction == 1 else "No Burnout", f"{confidence:.2f}%"

# # Define Gradio inputs and outputs
# inputs = [
#     gr.inputs.Slider(18, 60, step=1, label="Age"),
#     gr.inputs.Slider(100, 1500, step=50, label="Daily Rate"),
#     gr.inputs.Slider(1, 30, step=1, label="Distance From Home"),
#     gr.inputs.Radio(["Male", "Female"], label="Gender"),
#     gr.inputs.Slider(10, 100, step=5, label="Hourly Rate"),
#     gr.inputs.Slider(1000, 100000, step=1000, label="Monthly Income"),
#     gr.inputs.Slider(10000, 50000, step=500, label="Monthly Rate"),
#     gr.inputs.Slider(0, 10, step=1, label="Number of Companies Worked"),
#     gr.inputs.Radio(["Yes", "No"], label="Overtime"),
#     gr.inputs.Slider(0, 50, step=1, label="Percent Salary Hike"),
#     gr.inputs.Slider(1, 5, step=1, label="Performance Rating"),
#     gr.inputs.Slider(0, 5, step=1, label="Stock Option Level"),
#     gr.inputs.Slider(0, 40, step=1, label="Total Working Years"),
#     gr.inputs.Slider(0, 10, step=1, label="Training Times Last Year"),
#     gr.inputs.Slider(0, 40, step=1, label="Years at Company"),
#     gr.inputs.Slider(0, 20, step=1, label="Years in Current Role"),
#     gr.inputs.Slider(0, 15, step=1, label="Years Since Last Promotion"),
#     gr.inputs.Slider(0, 20, step=1, label="Years with Current Manager"),
#     gr.inputs.Radio(["Travel_Frequently", "Travel_Rarely", "Non-Travel"], label="Business Travel"),
#     gr.inputs.Radio(["Research & Development", "Sales", "Other"], label="Department"),
#     gr.inputs.Radio(["Below College", "College", "Doctor", "Master"], label="Education"),
#     gr.inputs.Radio(["Life Sciences", "Marketing", "Medical", "Other", "Technical Degree"], label="Education Field"),
#     gr.inputs.Radio(["Low", "Medium", "Very High"], label="Environment Satisfaction"),
#     gr.inputs.Radio(["Low", "Medium", "Very High"], label="Job Involvement"),
#     gr.inputs.Radio(["Executive Level", "Junior Level", "Mid Level", "Senior Level"], label="Job Level"),
#     gr.inputs.Radio(["Human Resources", "Laboratory Technician", "Manager", "Manufacturing Director", "Research Director", "Research Scientist", "Sales Executive", "Sales Representative"], label="Job Role"),
#     gr.inputs.Radio(["Married", "Single", "Divorced"], label="Marital Status"),
#     gr.inputs.Radio(["Low", "Medium", "Very High"], label="Relationship Satisfaction"),
#     gr.inputs.Radio(["Best", "Better", "Good"], label="Work Life Balance")
# ]

# outputs = [
#     gr.outputs.Textbox(label="Burnout Prediction"),
#     gr.outputs.Textbox(label="Confidence Level")
# ]

# # Create Gradio interface
# gr.Interface(
#     fn=predict_burnout, 
#     inputs=inputs, 
#     outputs=outputs, 
#     live=True
# ).launch()
import gradio as gr
import numpy as np
import pickle

# Load the model
with open("model.pkl", "rb") as f:
    model = pickle.load(f)

# Define the encoding function (as shared earlier)
def encode_input(
    age, daily_rate, distance_from_home, gender, hourly_rate, monthly_income,
    monthly_rate, num_companies_worked, overtime, percent_salary_hike, performance_rating,
    stock_option_level, total_working_years, training_times_last_year, years_at_company,
    years_in_current_role, years_since_last_promotion, years_with_curr_manager,
    business_travel, department, education, education_field, environment_satisfaction,
    job_involvement, job_level, job_role, job_satisfaction, marital_status,
    relationship_satisfaction, work_life_balance
):
    # Binary and numerical variables
    gender_binary = 1 if gender == "Male" else 0
    overtime_binary = 1 if overtime == "Yes" else 0

    # One-hot encodings (mapping for the relevant categories)
    business_travel_one_hot = {
        "Travel_Frequently": [1, 0],
        "Travel_Rarely": [0, 1],
        "Non-Travel": [0, 0],
    }
    department_one_hot = {
        "Research & Development": [1, 0],
        "Sales": [0, 1],
        "HR": [0, 0],
    }
    education_one_hot = {
        "Below College": [1, 0, 0, 0],
        "College": [0, 1, 0, 0],
        "Doctor": [0, 0, 1, 0],
        "Master": [0, 0, 0, 1],
    }
    education_field_one_hot = {
        "Life Sciences": [1, 0, 0, 0, 0],
        "Marketing": [0, 1, 0, 0, 0],
        "Medical": [0, 0, 1, 0, 0],
        "Other": [0, 0, 0, 1, 0],
        "Technical Degree": [0, 0, 0, 0, 1],
    }
    environment_satisfaction_one_hot = {
        "Low": [1, 0, 0],
        "Medium": [0, 1, 0],
        "Very High": [0, 0, 1],
    }
    job_involvement_one_hot = {
        "Low": [1, 0, 0],
        "Medium": [0, 1, 0],
        "Very High": [0, 0, 1],
    }
    job_level_one_hot = {
        "Executive Level": [1, 0, 0, 0],
        "Junior Level": [0, 1, 0, 0],
        "Mid Level": [0, 0, 1, 0],
        "Senior Level": [0, 0, 0, 1],
    }
    job_role_one_hot = {
        "Human Resources": [1, 0, 0, 0, 0, 0, 0, 0],
        "Laboratory Technician": [0, 1, 0, 0, 0, 0, 0, 0],
        "Manager": [0, 0, 1, 0, 0, 0, 0, 0],
        "Manufacturing Director": [0, 0, 0, 1, 0, 0, 0, 0],
        "Research Director": [0, 0, 0, 0, 1, 0, 0, 0],
        "Research Scientist": [0, 0, 0, 0, 0, 1, 0, 0],
        "Sales Executive": [0, 0, 0, 0, 0, 0, 1, 0],
        "Sales Representative": [0, 0, 0, 0, 0, 0, 0, 1],
    }
    job_satisfaction_one_hot = {
        "Low": [1, 0, 0],
        "Medium": [0, 1, 0],
        "Very High": [0, 0, 1],
    }
    marital_status_one_hot = {
        "Married": [1, 0],
        "Single": [0, 1],
        "Divorced": [0, 0],
    }
    relationship_satisfaction_one_hot = {
        "Low": [1, 0, 0],
        "Medium": [0, 1, 0],
        "Very High": [0, 0, 1],
    }
    work_life_balance_one_hot = {
        "Best": [1, 0, 0],
        "Better": [0, 1, 0],
        "Good": [0, 0, 1],
    }

    # Combine all features into a list
    encoded_data = [
        age, daily_rate, distance_from_home, gender_binary, hourly_rate,
        monthly_income, monthly_rate, num_companies_worked, overtime_binary,
        percent_salary_hike, performance_rating, stock_option_level,
        total_working_years, training_times_last_year, years_at_company,
        years_in_current_role, years_since_last_promotion, years_with_curr_manager
    ] + business_travel_one_hot[business_travel] \
      + department_one_hot[department] \
      + education_one_hot[education] \
      + education_field_one_hot[education_field] \
      + environment_satisfaction_one_hot[environment_satisfaction] \
      + job_involvement_one_hot[job_involvement] \
      + job_level_one_hot[job_level] \
      + job_role_one_hot[job_role] \
      + job_satisfaction_one_hot[job_satisfaction] \
      + marital_status_one_hot[marital_status] \
      + relationship_satisfaction_one_hot[relationship_satisfaction] \
      + work_life_balance_one_hot[work_life_balance]

    return np.array(encoded_data)

# Define the prediction function
def predict_burnout(age, daily_rate, distance_from_home, gender, hourly_rate, monthly_income,
                    monthly_rate, num_companies_worked, overtime, percent_salary_hike, performance_rating,
                    stock_option_level, total_working_years, training_times_last_year, years_at_company,
                    years_in_current_role, years_since_last_promotion, years_with_curr_manager,
                    business_travel, department, education, education_field, environment_satisfaction,
                    job_involvement, job_level, job_role, job_satisfaction, marital_status,
                    relationship_satisfaction, work_life_balance):
    input_data = encode_input(
        age, daily_rate, distance_from_home, gender, hourly_rate, monthly_income,
        monthly_rate, num_companies_worked, overtime, percent_salary_hike, performance_rating,
        stock_option_level, total_working_years, training_times_last_year, years_at_company,
        years_in_current_role, years_since_last_promotion, years_with_curr_manager,
        business_travel, department, education, education_field, environment_satisfaction,
        job_involvement, job_level, job_role, job_satisfaction, marital_status,
        relationship_satisfaction, work_life_balance
    )
    prediction = model.predict([input_data])
    confidence = model.predict_proba([input_data])[0][prediction[0]] * 100
    return f"Prediction: {'Burnout' if prediction[0] == 1 else 'No Burnout'}, Confidence: {confidence:.2f}%"

# Create the Gradio interface
inputs = [
    gr.inputs.Slider(18, 60, step=1, label="Age"),
    gr.inputs.Slider(0, 2000, step=1, label="Daily Rate"),
    gr.inputs.Slider(0, 50, step=1, label="Distance From Home"),
    gr.inputs.Dropdown(["Male", "Female"], label="Gender"),
    gr.inputs.Slider(0, 200, step=1, label="Hourly Rate"),
    gr.inputs.Slider(1000, 20000, step=1, label="Monthly Income"),
    gr.inputs.Slider(0, 50000, step=1, label="Monthly Rate"),
    gr.inputs.Slider(0, 10, step=1, label="Num Companies Worked"),
    gr.inputs.Dropdown(["Yes", "No"], label="Overtime"),
    gr.inputs.Slider(0, 50, step=1, label="Percent Salary Hike"),
    gr.inputs.Slider(1, 5, step=1, label="Performance Rating"),
    gr.inputs.Slider(0, 4, step=1, label="Stock Option Level"),
    gr.inputs.Slider(0, 40, step=1, label="Total Working Years"),
    gr.inputs.Slider(0, 10, step=1, label="Training Times Last Year"),
    gr.inputs.Slider(0, 40, step=1, label="Years At Company"),
    gr.inputs.Slider(0, 20, step=1, label="Years In Current Role"),
    gr.inputs.Slider(0, 10, step=1, label="Years Since Last Promotion"),
    gr.inputs.Slider(0, 20, step=1, label="Years With Current Manager"),
    gr.inputs.Dropdown(["Travel_Frequently", "Travel_Rarely", "Non-Travel"], label="Business Travel"),
    gr.inputs.Dropdown(["Research & Development", "Sales", "HR"], label="Department"),
    gr.inputs.Dropdown(["Below College", "College", "Doctor", "Master"], label="Education"),
    gr.inputs.Dropdown(["Life Sciences", "Marketing", "Medical", "Other", "Technical Degree"], label="Education Field"),
    gr.inputs.Dropdown(["Low", "Medium", "Very High"], label="Environment Satisfaction"),
    gr.inputs.Dropdown(["Low", "Medium", "Very High"], label="Job Involvement"),
    gr.inputs.Dropdown(["Executive Level", "Junior Level", "Mid Level", "Senior Level"], label="Job Level"),
    gr.inputs.Dropdown(["Human Resources", "Laboratory Technician", "Manager", "Manufacturing Director", "Research Director", "Research Scientist", "Sales Executive", "Sales Representative"], label="Job Role"),
    gr.inputs.Dropdown(["Low", "Medium", "Very High"], label="Job Satisfaction"),
    gr.inputs.Dropdown(["Married", "Single", "Divorced"], label="Marital Status"),
    gr.inputs.Dropdown(["Low", "Medium", "Very High"], label="Relationship Satisfaction"),
    gr.inputs.Dropdown(["Best", "Better", "Good"], label="Work Life Balance")
]

output = gr.outputs.Textbox(label="Burnout Prediction")

# Launch the Gradio interface
gr.Interface(fn=predict_burnout, inputs=inputs, outputs=output, live=True, allow_flagging="never", title="Burnout Prediction Model", description="Fill the fields and click 'Predict' to see the result.").launch(share=False)


