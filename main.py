# # import gradio as gr
import pickle
import numpy as np

# # Load the trained model
with open("model.pkl", "rb") as f:
    model = pickle.load(f)
import numpy as np

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

    # One-hot encodings
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

    # Combine all features
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

# Prediction function
def predict_burnout(age, monthly_income, department, education, environment_satisfaction, job_level, job_role, job_satisfaction, marital_status, overtime, business_travel_frequency, percent_salary_hike, stock_option_level, years_at_company, years_since_last_promotion):
    input_data = encode_input(age, monthly_income, department, education, environment_satisfaction, job_level, job_role, job_satisfaction, marital_status, overtime, business_travel_frequency, percent_salary_hike, stock_option_level, years_at_company, years_since_last_promotion)
    input_data = input_data.reshape(1, -1)
    prediction = model.predict(input_data)
    confidence = model.predict_proba(input_data)[0][prediction[0]] * 100
    return "Burnout" if prediction[0] == 1 else "No Burnout", confidence

test_input = encode_input(
    age=30, daily_rate=500, distance_from_home=10, gender="Male",
    hourly_rate=50, monthly_income=5000, monthly_rate=20000,
    num_companies_worked=2, overtime="Yes", percent_salary_hike=15,
    performance_rating=3, stock_option_level=1, total_working_years=10,
    training_times_last_year=2, years_at_company=5, years_in_current_role=3,
    years_since_last_promotion=1, years_with_curr_manager=2,
    business_travel="Travel_Frequently", department="Sales",
    education="College", education_field="Life Sciences",
    environment_satisfaction="Medium", job_involvement="Medium",
    job_level="Mid Level", job_role="Sales Executive",
    job_satisfaction="Medium", marital_status="Single",
    relationship_satisfaction="Medium", work_life_balance="Better"
)
print(len(test_input))  # Should print 60

prediction = model.predict([test_input])
confidence = model.predict_proba([test_input])[0][prediction[0]] * 100
print(f"Prediction: {'Burnout' if prediction[0] == 1 else 'No Burnout'}, Confidence: {confidence:.2f}%")


print('hello')

