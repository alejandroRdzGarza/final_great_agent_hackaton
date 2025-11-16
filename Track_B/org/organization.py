# org/organization.py
from typing import List, Dict
from .department import Department
class Organization:
    def __init__(self, name: str):
        self.name = name
        self.departments: Dict[str, Department] = {}

    def add_department(self, dept_name: str, supervisor=None):
        dept = Department(dept_name, supervisor)
        self.departments[dept_name] = dept
        return dept

    def get_all_agents(self):
        agents = []
        for dept in self.departments.values():
            agents.extend(dept.get_all_agents())
        return agents