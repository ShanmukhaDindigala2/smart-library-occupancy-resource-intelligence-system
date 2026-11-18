import pandas as pd


def recommend_study_area(occupancy_percentage):

    if occupancy_percentage >= 85:

        return [
            "Discussion Room",
            "Digital Study Area",
            "Reading Hall - Block B"
        ]

    elif occupancy_percentage >= 70:

        return [
            "Digital Study Area",
            "Reference Section"
        ]

    else:

        return [
            "Main Reading Hall",
            "Reference Section"
        ]


def recommend_resources(resource_name):

    alternatives = {
        "Python Programming": [
            "Data Structures",
            "Web Development"
        ],

        "Machine Learning": [
            "Artificial Intelligence",
            "Data Science"
        ],

        "Data Structures": [
            "Algorithms",
            "Programming Fundamentals"
        ],

        "Computer Networks": [
            "Operating Systems",
            "Cloud Computing"
        ],

        "Operating Systems": [
            "Computer Networks",
            "Software Engineering"
        ]
    }

    return alternatives.get(
        resource_name,
        [
            "Check Digital Library",
            "Check Reference Section"
        ]
    )


def get_resource_demand_level(demand):

    if demand >= 60:
        return "Very High"

    elif demand >= 40:
        return "High"

    elif demand >= 20:
        return "Moderate"

    else:
        return "Low"


if __name__ == "__main__":

    print(
        recommend_study_area(90)
    )

    print(
        recommend_resources(
            "Machine Learning"
        )
    )