import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import seaborn as sns
import streamlit as st
from PIL import Image
from plotly.subplots import make_subplots

st.set_page_config(
    page_title="GetAround Dashboard",
    layout="wide"
)

st.title("GetAround Dashboard")

def load_data():
    df = pd.read_excel('get_around_delay_analysis.xlsx')
    return df

df = load_data()
# Adding a column for delayed rentals
df["delay_status"] = df["delay_at_checkout_in_minutes"] > 0
# replace True and False by late and on time
df["delay_status"] = df["delay_status"].replace({True: "late", False: "on time"})

# removing outliers
# df = df[(df["delay_at_checkout_in_minutes"] < 720) & (df["delay_at_checkout_in_minutes"] > -720)]

st.markdown("## Main KPIs")

col1,col2,col3,col4, col5 = st.columns(5) # KPIs

tmp = round((df[df["state"] == "canceled"].shape[0] / df.shape[0]) * 100, 1)
col1.metric("Total rentals", df.shape[0], delta=f"{str(tmp)}% canceled", delta_color="inverse")
col2.metric("Total delay_status rentals", df[df["delay_status"] == "late"].shape[0])
col3.metric("Total delay_status rentals (%)", round(df[df["delay_status"] == "late"].shape[0] / df.shape[0] * 100, 0))
col4.metric("Total on time rentals", df[df["delay_status"] == "on time"].shape[0])
col5.metric("Total on time rentals (%)", round(df[df["delay_status"] == "on time"].shape[0] / df.shape[0] * 100, 0))

st.markdown("## Rentals Analysis")

col1,col2,col3,col4 = st.columns((1, 1, 1, 2)) # rental analysis

with col1: # state distribution
    tmp = df['state'].value_counts()
    fig = px.pie(values=tmp.values, names=tmp.index, title='State distribution')
    st.plotly_chart(fig, use_container_width=True)

with col2: # checkin type distribution
    tmp = df['checkin_type'].value_counts()
    fig = px.pie(values=tmp.values, names=tmp.index, title='Checkin type distribution')
    st.plotly_chart(fig, use_container_width=True)

# Proportion delay_status per rental type
mask = (df['delay_status'] == "late") & ((df['delay_at_checkout_in_minutes'] < 720) & (df['delay_at_checkout_in_minutes'] > -720))
df_tmp = df[mask]
tmp2 = df_tmp.groupby(df_tmp['checkin_type'])["delay_at_checkout_in_minutes"].count()
tmp3 = df_tmp.groupby(df_tmp['state'])["delay_at_checkout_in_minutes"].count()

with col3: # proportion of late checkout
    fig = px.pie(values=tmp2.values, names=tmp2.index, title='Proportion of late checkout per type')
    st.plotly_chart(fig, use_container_width=True)

with col4: # distribution of delay at checkout
    data = df.dropna(subset=['delay_at_checkout_in_minutes'])

    # Filter data by checkin_type
    all_data = data['delay_at_checkout_in_minutes']
    mobile_data = data[data['checkin_type'] == 'mobile']['delay_at_checkout_in_minutes']
    connect_data = data[data['checkin_type'] == 'connect']['delay_at_checkout_in_minutes']

    # Calculate average delay for each checkin_type
    average_all = all_data.mean()
    average_mobile = mobile_data.mean()
    average_connect = connect_data.mean()

    fig = go.Figure()

    fig.add_trace(go.Histogram(x=all_data, name="All", marker_color="lightsalmon"))
    fig.add_trace(go.Histogram(x=mobile_data, name="Mobile", visible=False, marker_color="lightblue"))
    fig.add_trace(go.Histogram(x=connect_data, name="Connect", visible=False, marker_color="lightgreen"))

    fig.update_layout(
        title="Distribution of delay at checkout",
        xaxis_title="Delay checkout in minutes",
        xaxis=dict(range=[-600, 600]),
        updatemenus=[
            dict(
                type="buttons",
                showactive=True,
                x=1,
                xanchor="right",
                y=1.2,
                yanchor="top",
                buttons=list([
                    dict(
                        label="All",
                        method="update",
                        args=[{"visible": [True, False, False]}, {"title": "Distribution of delay at checkout (All)", "xaxis": {"title": "Delay checkout in minutes", "range": [-600, 600]}}]),
                    dict(
                        label="Mobile",
                        method="update",
                        args=[{"visible": [False, True, False]}, {"title": "Distribution of delay at checkout (Mobile)", "xaxis": {"title": "Delay checkout in minutes", "range": [-600, 600]}}]),
                    dict(
                        label="Connect",
                        method="update",
                        args=[{"visible": [False, False, True]}, {"title": "Distribution of delay at checkout (Connect)", "xaxis": {"title": "Delay checkout in minutes", "range": [-600, 600]}}]),
                ]),
            )
        ],
    )

    st.plotly_chart(fig, use_container_width=True)

col1,col2,col3,col4 = st.columns((1, 1, 1, 2)) # rental analysis text
with col1:
    st.markdown("15% of the rentals are canceled")
with col2:
    st.markdown("80% of users are using mobile  \ncheckin")
with col3:
    st.markdown("mobile checkin are  \nmore likely to be late")
with col4:
    st.markdown("the average delay at checkout is lower for  \nconnect checkin")

col1, col2 = st.columns((2.5, 2.5)) # proportion of ended_canceled by checking type

with col1:
    # group data
    grouped_data_state = df.groupby(['checkin_type', 'state']).size().reset_index(name='count')
    grouped_data_delay_status = df.groupby(['checkin_type', 'delay_status']).size().reset_index(name='count')

    # calculating count
    total_count = df['checkin_type'].value_counts().reset_index()
    total_count.columns = ['checkin_type', 'total']

    # % for state
    grouped_data_state = grouped_data_state.merge(total_count, on='checkin_type')
    grouped_data_state['percentage'] = grouped_data_state['count'] / grouped_data_state['total'] * 100

    # % for for delay_status
    grouped_data_delay_status = grouped_data_delay_status.merge(total_count, on='checkin_type')
    grouped_data_delay_status['percentage'] = grouped_data_delay_status['count'] / grouped_data_delay_status['total'] * 100
    
    fig_state = go.Figure()

    fig_state.add_trace(go.Bar(
        x=grouped_data_state[grouped_data_state['state'] == 'canceled']['checkin_type'],
        y=grouped_data_state[grouped_data_state['state'] == 'canceled']['percentage'],
        name='Canceled',
        text=grouped_data_state[grouped_data_state['state'] == 'canceled']['percentage'].round(1).astype(str) + '%',
        textposition='inside',
        marker_color='indianred'
    ))

    fig_state.add_trace(go.Bar(
        x=grouped_data_state[grouped_data_state['state'] == 'ended']['checkin_type'],
        y=grouped_data_state[grouped_data_state['state'] == 'ended']['percentage'],
        name='Ended',
        text=grouped_data_state[grouped_data_state['state'] == 'ended']['percentage'].round(1).astype(str) + '%',
        textposition='inside',
        marker_color='lightsalmon'
    ))

    fig_state.update_layout(
        title='Proportion of state (ended/canceled) by checkin type',
        barmode='stack'
    )

    st.plotly_chart(fig_state, use_container_width=True)

with col2: # proportion of delay status by checkin type
    fig_delay_status = go.Figure()

    fig_delay_status.add_trace(go.Bar(
        x=grouped_data_delay_status[grouped_data_delay_status['delay_status'] == 'late']['checkin_type'],
        y=grouped_data_delay_status[grouped_data_delay_status['delay_status'] == 'late']['percentage'],
        name='Late',
        text=grouped_data_delay_status[grouped_data_delay_status['delay_status'] == 'late']['percentage'].round(1).astype(str) + '%',
        textposition='inside',
        marker_color='lightblue'
    ))

    fig_delay_status.add_trace(go.Bar(
        x=grouped_data_delay_status[grouped_data_delay_status['delay_status'] == 'on time']['checkin_type'],
        y=grouped_data_delay_status[grouped_data_delay_status['delay_status'] == 'on time']['percentage'],
        name='On Time',
        text=grouped_data_delay_status[grouped_data_delay_status['delay_status'] == 'on time']['percentage'].round(1).astype(str) + '%',
        textposition='inside',
        marker_color='deepskyblue'
    ))

    fig_delay_status.update_layout(
        title='Proportion of delay status (late/on time) by checkin',
        barmode='stack'
    )

    st.plotly_chart(fig_delay_status, use_container_width=True)

col1, col2 = st.columns((2.5, 2.5))
with col1:
    st.markdown("In proportion, the connect checkin rentals have more cancellation  \nbut there are less users using connect")
with col2:
    st.markdown("In proportion, the mobile checkin rentals are more likely to be late  \nbut there are also more users using mobile")
    st.markdown("We can't see a direct correlation between the checkin type and the cancellation rate")

# st.markdown("##### Average delays per checkin type (all rentals)") ## average delays 

# # Display average delays (entire dataset)
# # st.markdown(f"Average delay (All): {average_all:.2f} minutes")
# st.markdown(f"Average delay (Mobile): {average_mobile:.2f} minutes")
# st.markdown(f"Average delay (Connect): {average_connect:.2f} minutes")

# Calculate average delay when delay_status is late
all_data_late = data[data['delay_status'] == 'late']['delay_at_checkout_in_minutes'].mean()
mobile_data_late = data[data['delay_status'] == 'late'][data['checkin_type'] == 'mobile']['delay_at_checkout_in_minutes'].mean()
connect_data_late = data[data['delay_status'] == 'late'][data['checkin_type'] == 'connect']['delay_at_checkout_in_minutes'].mean()

# st.markdown("##### Average delays per checkin type (late rentals)")
# # st.markdown(f"Average delay (All): {all_data_late:.2f} minutes")
# st.markdown(f"Average delay (Mobile): {mobile_data_late:.2f} minutes")
# st.markdown(f"Average delay (Connect): {connect_data_late:.2f} minutes")

st.title("Cars analysis") 

col1, col2, col3 = st.columns((1.5, 1.5, 1.5))

with col1: # amount of rentals per count of cars
    car_rental_counts = df['car_id'].value_counts().reset_index()
    car_rental_counts.columns = ['car_id', 'count']

    fig = go.Figure(go.Histogram(x=car_rental_counts['count'], nbinsx=50, marker_color="lightblue"))

    fig.update_layout(
        title="Number of times cars are rented",
        xaxis_title="Number of Rentals",
        yaxis_title="Count of Cars",
        xaxis=dict(range=[0, 15]),
    )
    st.plotly_chart(fig, use_container_width=True)

with col2: # proportion of cars per checkin type
    car_id_counts = df['car_id'].value_counts().reset_index()
    car_id_counts.columns = ['car_id', 'count']

    merged_df = pd.merge(df, car_id_counts, on='car_id')
    merged_grouped = merged_df.groupby('checkin_type').agg({'car_id': 'nunique'}).reset_index()

    # Create the pie chart
    fig = go.Figure(go.Pie(labels=merged_grouped['checkin_type'], values=merged_grouped['car_id'], hole=.4))

    # Update the layout
    fig.update_layout(
        title="Proportion of cars per checkin type (all rentals)"
    )
    st.plotly_chart(fig, use_container_width=True)

with col3:
    def create_pie(threshold):
        filtered_cars = car_rental_counts[car_rental_counts['count'] > threshold]['car_id']
        filtered_df = df[df['car_id'].isin(filtered_cars)]
        grouped_filtered = filtered_df.groupby('checkin_type').agg({'car_id': 'nunique'}).reset_index()
        return go.Pie(labels=grouped_filtered['checkin_type'], values=grouped_filtered['car_id'], hole=.4)

    fig = go.Figure()

    for i, threshold in enumerate(range(1, 6)):
        pie_trace = create_pie(threshold)
        pie_trace.visible = i == 0  
        fig.add_trace(pie_trace)

    fig.update_layout(
        updatemenus=[
            go.layout.Updatemenu(
                buttons=[
                    dict(
                        label=f"Threshold: {threshold}",
                        method="update",
                        args=[{"visible": [i == threshold - 1 for i in range(len(fig.data))]}],
                    )
                    for threshold in range(1, 6)
                ],
                showactive=True,
                direction="down",
                pad={"r": 10, "t": 10},
                x=0.15,
                xanchor="left",
                y=1.1,
                yanchor="top",
            ),
        ],
        title_text="Proportion of cars rented more than (threshold) time(s) per checkin type",
    )
    st.plotly_chart(fig, use_container_width=True)

st.markdown("- By analyzing the car_id, we can see that some cars are rented more than once (obviously) but even if connect cars only represent 9% of the total rentals, they represent 14% of the cars rented more than once.")
st.markdown("- This ratio goes up to 39% for cars rented 5 times. This is a good indicator that connect cars are more likely to be rented more than once.")
st.markdown("- We can suppose that the users who rent connect cars are more likely to be regular users, connect is more likely to be used by users who are more familiar with the service\
 and/or that connect makes it easier to rent a car.")

st.markdown("## Analysis of available previous rentals")

merged_data = df.merge(df, left_on='previous_ended_rental_id', right_on='rental_id', suffixes=(None, '_previous'))
merged_data = merged_data.dropna(subset=['time_delta_with_previous_rental_in_minutes'])

st.markdown(f"Number of rentals with previous rental info provided: {len(merged_data)}")

col1, col2, col3 = st.columns((2, 2, 1)) # merged data

with col1: #boxplot 
    mean_value = merged_data['time_delta_with_previous_rental_in_minutes'].mean()
    median_value = merged_data['time_delta_with_previous_rental_in_minutes'].median()

    fig = go.Figure()
    fig.add_trace(go.Box(x=merged_data['time_delta_with_previous_rental_in_minutes'], marker_color='lightblue', boxmean=True))

    fig.add_shape(type='line', x0=mean_value, x1=mean_value, y0=0, y1=1, yref='paper', line=dict(color='blue', width=2, dash='dot'))
    fig.add_shape(type='line', x0=median_value, x1=median_value, y0=0, y1=1, yref='paper', line=dict(color='green', width=2, dash='dot'))

    fig.update_layout(title='Distribution of time delta with previous rental',
                    xaxis_title='Time Delta with Previous Rental (minutes)',
                    annotations=[
                        dict(x=mean_value, y=0.5, yref='paper', xanchor='left', text='Mean', showarrow=False, font=dict(color='blue')),
                        dict(x=median_value, y=0.5, yref='paper', xanchor='right', text='Median', showarrow=False, font=dict(color='orange'))
                    ],
                    template='plotly_white')
    st.plotly_chart(fig, use_container_width=True)

merged_data['checkout'] = np.where(merged_data['delay_at_checkout_in_minutes_previous'] > 0, 'late', 'on time')

canceled_data = merged_data[(merged_data['state'] == 'canceled')]

with col2: # number of late and on-time in merge_data
    canceled_counts = canceled_data['checkout'].value_counts()
    fig = go.Figure(go.Pie(labels=canceled_counts.index, values=canceled_counts, textinfo='label+value'))
    fig.update_layout(title='Canceled rentals with previous delay at checkout')
    st.plotly_chart(fig, use_container_width=True)

with col3: # proportion of late
    previous_delay_canceled = merged_data[(merged_data['delay_at_checkout_in_minutes_previous'] >= 0) & (merged_data['state'] == 'canceled')]

    # Count the occurrences of each checkin_type
    checkin_counts = previous_delay_canceled['checkin_type'].value_counts()

    # Calculate the proportion of each checkin_type
    checkin_proportions = checkin_counts / checkin_counts.sum()

    # Create the bar chart with proportions
    fig = go.Figure(data=[
        go.Bar(name='Proportion', x=checkin_proportions.index, y=checkin_counts.values, marker_color='lightblue', text=checkin_counts, textposition='auto')
    ])

    # Update the layout
    fig.update_layout(
        title='Amount of late rentals per checking type',
        xaxis_title='Checkin Type',
        yaxis_title='Proportion'
    )

    st.plotly_chart(fig, use_container_width=True)

col1, col2, = st.columns((2, 3)) # merged data
with col1:
    st.markdown("Threshold that we will use: 180 which is the median of the time delta with previous rental")
with col2: 
    st.markdown("Since the merged data is a subset of the original data, the amount of rentals with previous rental info provided is 1% of the dataset.")
    st.markdown("We can't draw conclusions or pursue more analysis on this data since it is too restricted.")
    st.markdown("There is also no pattern that we can see in the data since checkin_type proportion is the same")
    
threshold = df['delay_at_checkout_in_minutes'].median()

col1, col2, col3 = st.columns((2, 2, 1)) #jenémarre de faire des commentaires :()

with col1:
    df['delayed_with_threshold'] = df['delay_at_checkout_in_minutes'] < threshold
    delay_counts = df.groupby(['checkin_type', 'delayed_with_threshold']).size().unstack()

    fig = go.Figure()
    fig.add_trace(go.Bar(x=delay_counts.index, y=delay_counts[True], name='With Threshold'))
    fig.add_trace(go.Bar(x=delay_counts.index, y=delay_counts[False], name='Without Threshold'))
    fig.update_layout(title='Number of rentals delayed w/ & w/o Threshold', xaxis_title='Checkin Type', yaxis_title='Count', barmode='group')
    st.plotly_chart(fig, use_container_width=True)

with col2:
    # Group data by car_id and checkin_type, and count occurrences
    car_counts = df.groupby(['car_id', 'checkin_type']).agg({"delay_status": lambda x: (x == "late").sum()}).reset_index()
    car_counts.columns = ['car_id', 'checkin_type', 'late_count']

    # Create columns for never late, late once, and late multiple times
    car_counts['never_late'] = car_counts['late_count'] == 0
    car_counts['late_once'] = car_counts['late_count'] == 1
    car_counts['late_multiple_times'] = car_counts['late_count'] > 1

    # Group data by checkin_type and count the number of occurrences for each late status
    grouped_counts = car_counts.groupby('checkin_type')[['never_late', 'late_once', 'late_multiple_times']].sum()

    fig = go.Figure(data=[
        go.Bar(name='Never Late', x=grouped_counts.index, y=grouped_counts['never_late'], marker_color='lightsalmon'),
        go.Bar(name='Late Once', x=grouped_counts.index, y=grouped_counts['late_once'], marker_color='lightblue'),
        go.Bar(name='Late Multiple Times', x=grouped_counts.index, y=grouped_counts['late_multiple_times'], marker_color='lightgreen')
    ])

    fig.update_layout(
        barmode='group',
        title='Number of cars delayed or not checkouts',
        xaxis_title='Checkin type',
        yaxis_title='Number of cars'
    )
    st.plotly_chart(fig, use_container_width=True)
    
# with col2:
#     late_checkouts = df[df['delay_at_checkout_in_minutes'] > 0]
#     car_counts = late_checkouts.groupby(['car_id', 'checkin_type']).size().reset_index(name='count')
#     car_counts['late_multiple_times'] = car_counts['count'] > 1
#     grouped_counts = car_counts.groupby(['late_multiple_times', 'checkin_type']).size().unstack(fill_value=0)

#     fig = go.Figure(data=[
#         go.Bar(name='Late Once', x=grouped_counts.columns, y=grouped_counts.loc[False], marker_color='lightblue'),
#         go.Bar(name='Late Multiple Times', x=grouped_counts.columns, y=grouped_counts.loc[True], marker_color='lightgreen')
#     ])

#     fig.update_layout(
#         barmode='group',
#         title='Number of cars late for checkouts (once vs multiple times)',
#         xaxis_title='Checkin type',
#         yaxis_title='Number of cars'
#     )

    # st.plotly_chart(fig, use_container_width=True)

with col3:
    late_checkouts = df[df['delay_at_checkout_in_minutes'] > 0]
    average_delay_time = late_checkouts.groupby('checkin_type')['delay_at_checkout_in_minutes'].mean()

    fig = go.Figure(data=[
        go.Bar(name='Average Delay Time', x=average_delay_time.index, y=average_delay_time.values, marker_color='lightblue')
    ])

    fig.update_layout(
        title='Average Delay Time for Each Checkin Type',
        xaxis_title='Checkin Type',
        yaxis_title='Average Delay Time (minutes)'
    )

    fig.show()
    st.plotly_chart(fig, use_container_width=True)

st.markdown("- The threshold would have a small negative impact on the connect cars since they will be delayed more than their average delay time.")
st.markdown("- In proportion, the connect cars are often late but the average delay is 80min")
col1, col2 = st.columns((2.5, 2.5))

with col1:
    late_checkouts = df[df['delay_at_checkout_in_minutes'] > 0]

    ranges = [(-1, 30), (30, 60), (60, 120), (120, 180), (180, float('inf'))]

    data_counts = []
    for lower, upper in ranges:
        data_counts.append(late_checkouts[(late_checkouts['delay_at_checkout_in_minutes'] > lower) & (late_checkouts['delay_at_checkout_in_minutes'] <= upper)]['checkin_type'].value_counts())

    counts_df = pd.concat(data_counts, axis=1).fillna(0)
    counts_df.columns = [f"{lower+1}-{upper} min" for lower, upper in ranges]

    fig = go.Figure()

    for col in counts_df.columns:
        fig.add_trace(go.Bar(name=col, x=counts_df.index, y=counts_df[col]))

    fig.update_layout(
        barmode='group',
        title='Number of rentals late within ranges',
        xaxis_title='Checkin type',
        yaxis_title='Number of rentals',
        updatemenus=[
            dict(
                type="buttons",
                showactive=True,
                buttons=list([
                    dict(
                        label="All",
                        method="update",
                        args=[{"visible": [True] * len(ranges) * 2}, {"title": "Number of rentals late within ranges (All)"}]),
                    dict(
                        label="Mobile",
                        method="update",
                        args=[{"visible": [True, False] * len(ranges)}, {"title": "Number of rentals late within ranges (Mobile)"}]),
                    dict(
                        label="Connect",
                        method="update",
                        args=[{"visible": [False, True] * len(ranges)}, {"title": "Number of rentals late within ranges  (Connect)"}])
                ]),
            )
        ],
    )
    
    # Set the initial visibility of the bars
    visibility_array = [True] * len(ranges) * 2
    fig.update_traces(visible=True, selector=dict(type='bar'))

    fig.show()

    st.plotly_chart(fig, use_container_width=True)

with col2:

    thresholds = [60, 120, 180, float('inf')]
    impacted_counts = [late_checkouts[late_checkouts['delay_at_checkout_in_minutes'] < t]['checkin_type'].value_counts() for t in thresholds[:-1]]
    impacted_counts.append(df[df['delay_at_checkout_in_minutes'] > 180]['checkin_type'].value_counts())

    impacted_df = pd.concat(impacted_counts, axis=1).fillna(0)
    impacted_df.columns = [f"<{t//60} hour(s)" if t != float('inf') else ">3 hours" for t in thresholds]

    fig = go.Figure()

    for col in impacted_df.columns:
        fig.add_trace(go.Bar(name=col, x=impacted_df.index, y=impacted_df[col]))

    fig.update_layout(
        barmode='group',
        title='Number of Rentals Impacted by Different Thresholds',
        xaxis_title='Checkin Type',
        yaxis_title='Number of Rentals'
    )

    st.plotly_chart(fig, use_container_width=True)

st.markdown("- We could conclude that using connect on the mobile cars would be an improvement to reduce the average delay time.")
st.markdown("- If I had to take a decision, I would put the threshold at 180min for mobile cars renting, it would affect ~6300 of rentals in this dataset.")
st.markdown("- Doing so would also incite the car owners to use connect instead of mobile.")
st.markdown("- Given the dataset provided, we can't have proper conclusions for the reasons of cancellations since we can't even know if a cancellation after the late checkout is the reason.")

