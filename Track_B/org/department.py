from typing import List, Dict

class Department:
    def __init__(self, name: str, supervisor=None):
        self.name = name
        self.agents = []
        self.supervisor = supervisor  # another agent
        self.sub_departments = []

    def add_agent(self, agent):
        self.agents.append(agent)

    def add_sub_department(self, department):
        self.sub_departments.append(department)

    def get_all_agents(self):
        agents = self.agents.copy()
        for sub in self.sub_departments:
            agents.extend(sub.get_all_agents())
        return agents