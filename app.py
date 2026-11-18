from pathlib import Path
import sys

import joblib
import pandas as pd
import streamlit as st


# ---------------------------------------------------------
# Project paths and imports
# ---------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent
SRC_DIR = BASE_DIR / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from seat_availability import (
    calculate_available_seats,
    occupancy_status,
    predict_study_area_seats,
    get_recommended_study_area,
)


st.set_page_config(
    page_title="Smart Library Occupancy & Resource Intelligence",
    page_icon="📚",
    layout="wide",
)

OCCUPANCY_MODEL_PATH = BASE_DIR / "models" / "occupancy_model.pkl"
OCCUPANCY_DATA_PATH = BASE_DIR / "data" / "occupancy_data.csv"


# ---------------------------------------------------------
# Book catalogue (sample data; replace with actual library data)
# ---------------------------------------------------------
BOOKS = [
    {"id": 1, "title": "Python Crash Course", "author": "Eric Matthes",
     "branch": "CSE", "course": "Python Programming", "semester": "1-2",
     "total": 10, "available": 6},
    {"id": 2, "title": "Introduction to Algorithms", "author": "Thomas H. Cormen",
     "branch": "CSE", "course": "Data Structures and Algorithms", "semester": "2-4",
     "total": 8, "available": 3},
    {"id": 3, "title": "Database System Concepts", "author": "Abraham Silberschatz",
     "branch": "CSE", "course": "Database Management Systems", "semester": "3-5",
     "total": 7, "available": 4},
    {"id": 4, "title": "Pattern Recognition and Machine Learning",
     "author": "Christopher Bishop", "branch": "CSE", "course": "Machine Learning",
     "semester": "5-8", "total": 5, "available": 2},
    {"id": 5, "title": "Computer Networking: A Top-Down Approach",
     "author": "James Kurose", "branch": "CSE", "course": "Computer Networks",
     "semester": "4-6", "total": 6, "available": 3},
    {"id": 6, "title": "Operating System Concepts", "author": "Abraham Silberschatz",
     "branch": "CSE", "course": "Operating Systems", "semester": "3-5",
     "total": 6, "available": 2},
    {"id": 7, "title": "Clean Code", "author": "Robert C. Martin",
     "branch": "IT", "course": "Software Engineering", "semester": "4-7",
     "total": 5, "available": 3},
    {"id": 8, "title": "Effective Java", "author": "Joshua Bloch",
     "branch": "IT", "course": "Java Programming", "semester": "2-5",
     "total": 6, "available": 4},
    {"id": 9, "title": "Digital Design", "author": "M. Morris Mano",
     "branch": "ECE", "course": "Digital Electronics", "semester": "2-4",
     "total": 7, "available": 4},
    {"id": 10, "title": "Microelectronic Circuits", "author": "Adel Sedra",
     "branch": "ECE", "course": "Electronic Circuits", "semester": "3-6",
     "total": 5, "available": 2},
    {"id": 11, "title": "Electric Machinery Fundamentals", "author": "Stephen Chapman",
     "branch": "EEE", "course": "Electrical Machines", "semester": "3-6",
     "total": 5, "available": 3},
    {"id": 12, "title": "Power System Analysis", "author": "Hadi Saadat",
     "branch": "EEE", "course": "Power Systems", "semester": "4-7",
     "total": 4, "available": 1},
    {"id": 13, "title": "Engineering Mechanics", "author": "R. C. Hibbeler",
     "branch": "Mechanical", "course": "Engineering Mechanics", "semester": "1-3",
     "total": 8, "available": 5},
    {"id": 14, "title": "Thermodynamics: An Engineering Approach",
     "author": "Yunus Cengel", "branch": "Mechanical", "course": "Thermodynamics",
     "semester": "3-5", "total": 6, "available": 2},
    {"id": 15, "title": "Surveying", "author": "B. C. Punmia",
     "branch": "Civil", "course": "Surveying", "semester": "2-4",
     "total": 6, "available": 4},
    {"id": 16, "title": "Design of Concrete Structures", "author": "M. S. Shetty",
     "branch": "Civil", "course": "Structural Engineering", "semester": "4-7",
     "total": 5, "available": 2},
    {"id": 17, "title": "Fundamentals of Chemical Engineering",
     "author": "David Himmelblau", "branch": "Chemical",
     "course": "Chemical Engineering", "semester": "2-6",
     "total": 5, "available": 3},
    {"id": 18, "title": "Engineering Mathematics", "author": "B. S. Grewal",
     "branch": "All Branches", "course": "Engineering Mathematics",
     "semester": "1-4", "total": 12, "available": 8},
    {"id": 19, "title": "Engineering Physics", "author": "H. K. Malik",
     "branch": "All Branches", "course": "Engineering Physics",
     "semester": "1-2", "total": 10, "available": 6},
    {"id": 20, "title": "Engineering Chemistry", "author": "Jain and Jain",
     "branch": "All Branches", "course": "Engineering Chemistry",
     "semester": "1-2", "total": 8, "available": 0},
]


# ---------------------------------------------------------
# Cached model and dataset loading
# ---------------------------------------------------------
@st.cache_resource
def load_model(path_string):
    return joblib.load(path_string)


@st.cache_data
def load_csv(path_string):
    return pd.read_csv(path_string)


# ---------------------------------------------------------
# Page heading
# ---------------------------------------------------------
st.title("📚 Smart Library Occupancy & Resource Intelligence")
st.write(
    "Predict library occupancy, compare study-area seat estimates, "
    "and search for books by branch and course."
)


# ---------------------------------------------------------
# Load and validate required occupancy files only
# ---------------------------------------------------------
missing_files = [
    str(path)
    for path in (OCCUPANCY_MODEL_PATH, OCCUPANCY_DATA_PATH)
    if not path.exists()
]

if missing_files:
    st.error("Required files were not found:")
    for missing in missing_files:
        st.write(f"- `{missing}`")
    st.info(
        "Check that the occupancy model and occupancy CSV exist. "
        "The book catalogue in this app uses sample data and does not "
        "require the demand model or resource-demand CSV."
    )
    st.stop()

try:
    occupancy_model = load_model(str(OCCUPANCY_MODEL_PATH))
    occupancy_df = load_csv(str(OCCUPANCY_DATA_PATH))
except Exception as error:
    st.error(f"Could not load the occupancy model or dataset: {error}")
    st.stop()

required_occupancy_columns = {
    "hour",
    "day_of_week",
    "is_weekend",
    "exam_period",
    "total_seats",
    "occupied_seats",
}

missing_columns = required_occupancy_columns.difference(
    occupancy_df.columns
)

if missing_columns:
    st.error(
        "The occupancy dataset is missing columns: "
        + ", ".join(sorted(missing_columns))
    )
    st.stop()


# ---------------------------------------------------------
# Sidebar prediction inputs
# ---------------------------------------------------------
st.sidebar.header("⚙️ Prediction Settings")

hour = st.sidebar.slider(
    "Select Hour",
    min_value=0,
    max_value=23,
    value=17,
    help="0 = midnight; 23 = 11 PM.",
)

day_names = [
    "Monday", "Tuesday", "Wednesday", "Thursday",
    "Friday", "Saturday", "Sunday",
]

day_name = st.sidebar.selectbox(
    "Select Day",
    day_names,
    index=0,
)

day_of_week = day_names.index(day_name)
is_weekend = int(day_of_week >= 5)
exam_period = int(st.sidebar.checkbox("Exam Period", value=False))

total_seats = int(occupancy_df["total_seats"].max())

st.sidebar.caption(f"Model library capacity: {total_seats} seats")


# ---------------------------------------------------------
# Overall occupancy prediction
# ---------------------------------------------------------
model_input = pd.DataFrame([{
    "hour": hour,
    "day_of_week": day_of_week,
    "is_weekend": is_weekend,
    "exam_period": exam_period,
    "total_seats": total_seats,
}])

try:
    predicted_occupied = float(occupancy_model.predict(model_input)[0])
except Exception as error:
    st.error(f"Occupancy prediction failed: {error}")
    st.stop()

predicted_occupied = max(
    0.0,
    min(float(total_seats), predicted_occupied),
)

filled_seats = int(round(predicted_occupied))
available_seats = calculate_available_seats(
    total_seats,
    predicted_occupied,
)
overall_percentage = (
    predicted_occupied / total_seats * 100
    if total_seats > 0
    else 0.0
)
status = occupancy_status(predicted_occupied, total_seats)


st.divider()
st.subheader("📊 Overall Library Prediction")

metric1, metric2, metric3, metric4 = st.columns(4)
metric1.metric("Total Seats", total_seats)
metric2.metric("Predicted Filled Seats", filled_seats)
metric3.metric("Predicted Empty Seats", available_seats)
metric4.metric("Occupancy", f"{overall_percentage:.1f}%")

st.progress(max(0.0, min(1.0, overall_percentage / 100)))
st.write(f"**Occupancy Status:** {status}")
st.caption(
    f"Selected conditions: {day_name}, {hour:02d}:00, "
    f"Exam period: {'Yes' if exam_period else 'No'}"
)


# ---------------------------------------------------------
# Study-area predictions
# ---------------------------------------------------------
st.divider()
st.subheader("📍 Recommended Study Areas")
st.write(
    "Compare estimated filled and empty seats for each study area. "
    "The area capacities and weights are configured in "
    "`src/seat_availability.py`."
)

try:
    area_predictions = predict_study_area_seats(overall_percentage)
except Exception as error:
    st.error(f"Study-area predictions failed: {error}")
    st.stop()

if area_predictions.empty:
    st.info("No study-area predictions are available.")
else:
    best_area = area_predictions.iloc[0]

    st.success(
        f"Suggested area: {best_area['Study Area']} — "
        f"{best_area['Predicted Empty Seats']} estimated empty seats."
    )

    area_col1, area_col2 = st.columns(2)
    area_col1.metric("Number of Study Areas", len(area_predictions))
    area_col2.metric(
        "Combined Listed Capacity",
        int(area_predictions["Total Seats"].sum()),
    )

    st.dataframe(
        area_predictions[
            [
                "Study Area",
                "Total Seats",
                "Predicted Filled Seats",
                "Predicted Empty Seats",
                "Occupancy (%)",
                "Status",
            ]
        ],
        use_container_width=True,
        hide_index=True,
    )

    st.caption(
        "Area-level figures are estimates based on configured weights, "
        "not separate trained predictions for each room. The listed "
        "capacities should be adjusted to match your real library."
    )


# ---------------------------------------------------------
# Book search and catalogue
# ---------------------------------------------------------
st.divider()
st.subheader("📚 Books & Resources")
st.write(
    "Search by book title, author, subject, or branch. Use the filters "
    "to narrow the results, then request a book directly from its result."
)

if "book_requests" not in st.session_state:
    st.session_state.book_requests = []

if "book_availability" not in st.session_state:
    st.session_state.book_availability = {
        book["id"]: book["available"] for book in BOOKS
    }

# The search input is the main control; matching books appear directly below.
search_text = st.text_input(
    "🔎 Search books",
    placeholder="Try 'machine', 'Python', an author's name, or a subject...",
    key="book_search",
).strip().casefold()

filter_col1, filter_col2, filter_col3 = st.columns(3)
branch_options = [
    "All Branches", "CSE", "IT", "ECE", "EEE",
    "Mechanical", "Civil", "Chemical",
]
with filter_col1:
    selected_branch = st.selectbox(
        "Branch", branch_options, key="book_branch"
    )

branch_books = [
    book for book in BOOKS
    if selected_branch == "All Branches"
    or book["branch"] in (selected_branch, "All Branches")
]
course_options = sorted({book["course"] for book in branch_books})
with filter_col2:
    selected_course = st.selectbox(
        "Course / Subject",
        ["All Courses"] + course_options,
        key="book_course",
    )
with filter_col3:
    availability_filter = st.selectbox(
        "Availability",
        ["All Books", "Available now", "Currently unavailable"],
        key="book_availability_filter",
    )

filtered_books = []
for book in branch_books:
    # When the user types a search, let the search term find matching
    # titles/authors/subjects even if a different course was selected earlier.
    # The course filter applies when the search box is empty.
    if (
        not search_text
        and selected_course != "All Courses"
        and book["course"] != selected_course
    ):
        continue

    available_copies = st.session_state.book_availability[book["id"]]
    if availability_filter == "Available now" and available_copies <= 0:
        continue
    if availability_filter == "Currently unavailable" and available_copies > 0:
        continue

    searchable_text = " ".join([
        book["title"], book["author"], book["course"],
        book["branch"], book["semester"],
    ]).casefold()
    # Support multi-word searches such as "machine learning" by matching
    # each word against the book's searchable fields.
    search_terms = search_text.split()
    if search_terms and not all(term in searchable_text for term in search_terms):
        continue
    filtered_books.append(book)

if search_text:
    st.caption(
        f"Search results for **{search_text}**. "
        "While searching, the search term takes priority over the Course / Subject filter."
    )

if not filtered_books:
    st.warning(
        "No books match your search and filters. Try a shorter search term "
        "or choose different filters."
    )
else:
    st.write(f"**{len(filtered_books)} matching book(s)**")

    for book in filtered_books:
        book_id = book["id"]
        available_copies = st.session_state.book_availability[book_id]
        total_copies = book["total"]
        borrowed_copies = max(0, total_copies - available_copies)

        with st.container(border=True):
            title_col, status_col = st.columns([4, 1])
            with title_col:
                st.markdown(f"#### 📖 {book['title']}")
                st.write(f"**Author:** {book['author']}")
                st.caption(
                    f"Branch: {book['branch']}  ·  Subject: {book['course']}  ·  "
                    f"Suggested semesters: {book['semester']}"
                )
            with status_col:
                if available_copies > 0:
                    st.success("Available")
                else:
                    st.warning("Unavailable")

            count_col1, count_col2, count_col3 = st.columns(3)
            count_col1.metric("Total copies", total_copies)
            count_col2.metric("Available", available_copies)
            count_col3.metric("Borrowed", borrowed_copies)

            if st.button(
                "📚 Request This Book" if available_copies > 0
                else "🕒 Request When Available",
                key=f"request_book_{book_id}",
                use_container_width=True,
            ):
                already_requested = any(
                    request["book_id"] == book_id
                    and request["status"] in ("Pending", "Waiting for availability")
                    for request in st.session_state.book_requests
                )
                if already_requested:
                    st.info("You already have an active request for this book.")
                else:
                    request_status = (
                        "Pending" if available_copies > 0
                        else "Waiting for availability"
                    )
                    st.session_state.book_requests.append({
                        "book_id": book_id,
                        "title": book["title"],
                        "branch": book["branch"],
                        "course": book["course"],
                        "status": request_status,
                    })
                    if request_status == "Pending":
                        st.success(
                            f"Request submitted for '{book['title']}'. Status: Pending. "
                            "This is a demo request saved for this app session only."
                        )
                    else:
                        st.warning(
                            f"'{book['title']}' is currently unavailable. "
                            "You have been added to the demo waiting list. "
                            "Status: Waiting for availability. This is saved for this app session only."
                        )

# ---------------------------------------------------------
# Current-session book requests
# ---------------------------------------------------------
st.divider()
st.subheader("📋 My Book Requests")

if st.session_state.book_requests:
    requests_df = pd.DataFrame(st.session_state.book_requests)
    requests_df = requests_df[
        ["title", "branch", "course", "status"]
    ].rename(columns={
        "title": "Book",
        "branch": "Branch",
        "course": "Course",
        "status": "Request Status",
    })

    st.dataframe(
        requests_df,
        use_container_width=True,
        hide_index=True,
    )
else:
    st.info("You have not submitted any book requests in this session.")


# ---------------------------------------------------------
# Historical occupancy chart - responds to selected conditions
# ---------------------------------------------------------
st.divider()
st.subheader("📈 Historical Occupancy by Hour")

if "occupancy_rate" in occupancy_df.columns:
    chart_df = occupancy_df.copy()

    # Match the selected day and exam-period setting so the graph
    # changes when the dashboard inputs change.
    chart_df = chart_df[
        (pd.to_numeric(chart_df["day_of_week"], errors="coerce") == day_of_week)
        & (pd.to_numeric(chart_df["exam_period"], errors="coerce") == exam_period)
    ]

    if chart_df.empty:
        st.info(
            "No historical rows match this day and exam-period setting. "
            "Showing all available historical data instead."
        )
        chart_df = occupancy_df.copy()

    # Average occupancy percentage for each hour under the selected conditions.
    hourly_history = (
        chart_df.groupby("hour", as_index=False)["occupancy_rate"]
        .mean()
        .sort_values("hour")
    )
    hourly_history["Hour"] = hourly_history["hour"].astype(int).map(
        lambda h: f"{h:02d}:00"
    )
    hourly_history = hourly_history.set_index("Hour")
    hourly_history = hourly_history.rename(
        columns={"occupancy_rate": "Average historical occupancy (%)"}
    )

    # Add a separate series with a value only at the selected hour.
    # This marks the selected hour on the chart without changing the historical values.
    selected_hour_label = f"{hour:02d}:00"
    hourly_history["Selected hour"] = float("nan")
    if selected_hour_label in hourly_history.index:
        hourly_history.loc[
            selected_hour_label, "Selected hour"
        ] = float(
            hourly_history.loc[
                selected_hour_label, "Average historical occupancy (%)"
            ]
        )

    st.caption(
        f"Historical averages for {day_name}; "
        f"exam period: {'Yes' if exam_period else 'No'}. "
        f"The selected hour is {selected_hour_label}. "
        "The line shows historical data; it is not a live occupancy sensor."
    )
    st.line_chart(
        hourly_history[
            ["Average historical occupancy (%)", "Selected hour"]
        ],
        y=["Average historical occupancy (%)", "Selected hour"],
        use_container_width=True,
    )

    selected_rows = chart_df[
        pd.to_numeric(chart_df["hour"], errors="coerce") == hour
    ]
    if not selected_rows.empty:
        historical_at_selected_hour = float(
            pd.to_numeric(
                selected_rows["occupancy_rate"], errors="coerce"
            ).mean()
        )
        st.metric(
            f"Historical average at {selected_hour_label}",
            f"{historical_at_selected_hour:.1f}%",
        )
    else:
        st.info("There is no historical value for the selected hour.")
else:
    st.info(
        "The historical chart is unavailable because "
        "`occupancy_rate` is missing from the occupancy dataset."
    )


st.caption(
    "This prototype uses generated occupancy data and a sample book "
    "catalogue. Per-area seat counts are estimates. Book availability "
    "and requests are not connected to a live library database."
)
