from flask import Flask, jsonify, request
from flask_sqlalchemy import SQLAlchemy
from faker import Faker
import random
from sklearn.linear_model import LinearRegression
import json
from sklearn.ensemble import RandomForestRegressor
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from flask_marshmallow import Marshmallow
from marshmallow_sqlalchemy import SQLAlchemyAutoSchema


scaler = StandardScaler()
import datetime

app = Flask(__name__)
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///employees.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
db = SQLAlchemy(app)
fake = Faker()
ma = Marshmallow(app)




# Employee Model
class Employee(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(50), nullable=False)
    department = db.Column(db.String(50), nullable=False)
    salary = db.Column(db.Float, nullable=False)
    hire_date = db.Column(db.DateTime, nullable=False, default=datetime.datetime.now)

    def __init__(self, name, department, salary, hire_date=None):
        self.name = name
        self.department = department
        self.salary = salary
        if hire_date is not None:
            self.hire_date = hire_date


class EmployeeSchema(SQLAlchemyAutoSchema):
    class Meta:
        model = Employee


employee_schema = EmployeeSchema()
employees_schema = EmployeeSchema(many=True)
# Initialize the scaler





# Generate fake employee data and insert into database
for i in range(1000):
    name = fake.name()
    department = fake.job()
    salary = round(random.uniform(0, 1000000), 2)
    hire_date = fake.date_time_between(start_date='-1y', end_date='now')
    employee = Employee(name, department, salary, hire_date)
    db.session.add(employee)
db.session.commit()

df = pd.read_sql_table('employee', 'sqlite:///employees.db')
X = df[['department', 'hire_date', 'name']]
y = df['salary']
X['hire_date'] = pd.to_datetime(X['hire_date']).astype(int) / 10**9
X = pd.get_dummies(X, columns=['department', 'name'])
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X, y)
# API endpoints
@app.route('/employees', methods=['GET'])
def get_employees():
    employees = Employee.query.all()
    result = employees_schema.dump(employees)
    return jsonify(result)


@app.route('/employees/<int:id>', methods=['GET'])
def get_employee(id):
    employee = Employee.query.get(id)
    if employee:
        return jsonify(employee.__dict__)
    else:
        return jsonify({'error': 'Employee not found'}), 404

@app.route('/employees', methods=['POST'])
def create_employee():
    data = request.get_json()
    if not data or not all(key in data for key in ['name', 'department', 'salary']):
        return jsonify({'error': 'Missing data'}), 400
    employee = Employee(data['name'], data['department'], data['salary'])
    db.session.add(employee)
    db.session.commit()
    return jsonify({'id': employee.id})

@app.route('/employees/<int:id>', methods=['PUT'])
def update_employee(id):
    employee = Employee.query.get(id)
    if not employee:
        return jsonify({'error': 'Employee not found'}), 404
    data = request.get_json()
    if not data:
        return jsonify({'error': 'No data provided'}), 400
    if 'name' in data:
        employee.name = data['name']
    if 'department' in data:
        employee.department = data['department']
    if 'salary' in data:
        employee.salary = data['salary']
    db.session.commit()
    return jsonify(employee.__dict__)

@app.route('/employees/<int:id>', methods=['DELETE'])
def delete_employee(id):
    employee = Employee.query.get(id)
    if not employee:
        return jsonify({'error': 'Employee not found'}), 404
    db.session.delete(employee)
    db.session.commit()
    return '', 204

@app.route('/departments', methods=['GET'])
def get_departments():
    departments = db.session.query(Employee.department).distinct().all()
    return jsonify([d[0] for d in departments])

@app.route('/departments/string:name')
def get_department_employees(name):
    employees = Employee.query.filter_by(department=name).all()
    if not employees:
        return jsonify({'message': 'No employees found in this department'}), 404
    else:
        result = employees_schema.dump(employees)
    return jsonify({'employees': result}), 200

@app.route('/average_salary/string:department')
def get_average_salary(department):
    average_salary = db.session.query(func.avg(Employee.salary)).filter_by(department=department).scalar()
    if not average_salary:
        return jsonify({'message': 'No employees found in this department'}), 404
    else:
        return jsonify({'average_salary': round(average_salary, 2)}), 200

@app.route('/top_earners')
def get_top_earners():
    employees = Employee.query.order_by(Employee.salary.desc()).limit(10).all()
    result = employees_schema.dump(employees)
    return jsonify({'employees': result}), 200

@app.route('/most_recent_hires')
def get_most_recent_hires():
    employees = Employee.query.order_by(Employee.hire_date.desc()).limit(10).all()
    result = employees_schema.dump(employees)
    return jsonify({'employees': result}), 200

#machine learning model endpoint
@app.route('/predict_salary', methods=['POST'])
def predict_salary():
    data = request.get_json()
    department = data['department']
    hire_date = datetime.strptime(data['hire_date'], '%Y-%m-%d %H:%M:%S')
    job_title = data['job_title']
    # preprocess the input data
    input_data = np.array([department, hire_date, job_title]).reshape(1, -1)
    input_data = scaler.transform(input_data)

# make the prediction using the pre-trained model
    predicted_salary = model.predict(input_data)

# return the predicted salary as a JSON response
    return jsonify({'predicted_salary': predicted_salary[0]})



if __name__ == '__main__':
    with app.app_context():
        
        db.create_all()
    app.run(debug=True)