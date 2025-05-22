import simpy
import random
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from tabulate import tabulate
from collections import defaultdict

SIMULATION_TIME = 10 * 60 # total sim time in minutes
VISITOR_ARRIVAL_MEAN = 0.5 # avg time bw visitor arrival
RIDE_NAMES = ["Carousel", "Slide Cars", "Race Cars", "Ferris Wheel",
              "Self-Control Planes", "Spiral Rides", "Flying Tower"]
RIDE_CAPACITY = {
    "Carousel": 2,
    "Slide Cars": 5,
    "Race Cars": 2,
    "Ferris Wheel": 10,
    "Self-Control Planes": 5,
    "Spiral Rides": 10,
    "Flying Tower": 3,
}
RIDE_SERVICE_TIME = {
    "Carousel": lambda: max(3, int(np.random.normal(10, 2))), #  most rides will last between 8 to 12 minutes, the max 3 ensures no ride is shorter than 3 minutes (mean, std)
    "Slide Cars": lambda: int(np.random.triangular(3, 5, 8)), #  most rides ;ast 5 mins, but could vary between 3 to 8 mins (mean, mode, max)
    "Race Cars": lambda: max(3, int(np.random.normal(12, 3))),
    "Ferris Wheel": lambda: max(1, int(np.random.normal(10, 3))),
    "Self-Control Planes": lambda: int(np.random.triangular(4, 6, 9)),
    "Spiral Rides": lambda: max(1, int(np.random.normal(6, 1))),
    "Flying Tower": lambda: int(np.random.triangular(10, 15, 20)),
}
RIDE_FAILURE_PROB = {
    "Carousel": 0.03,
    "Slide Cars": 0.01,
    "Race Cars": 0.03,
    "Ferris Wheel": 0.01,
    "Self-Control Planes": 0.01,
    "Spiral Rides": 0.01,
    "Flying Tower": 0.03,
}
RIDE_REPAIR_TIME = lambda: int(np.random.weibull(1.5) * 10)
VISITOR_PRIORITY = {
    "VIP": 0,
    "regular": 1,
}

event_logs_per_ride = {name: [] for name in RIDE_NAMES} # init list for each ride

def format_time(t):
    total_seconds = int(t * 60)
    hours = total_seconds // 3600
    minutes = (total_seconds % 3600) // 60
    seconds = total_seconds % 60
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"

class Ride:
    def __init__(self, env, name, capacity):
        self.env = env
        self.name = name
        self.capacity = capacity
        self.resource = simpy.PriorityResource(env, capacity) # simpy resource w priority
        self.broken = False
        self.total_visitors = 0
        self.dropped_visitors = 0
        self.total_wait_time = 0
        self.utilization_time = 0
        self.queue_lengths = []
        self.failures = 0
        self.completed_rides = 0
        self.state_durations = defaultdict(int)
        self.last_event_time = 0
        self.utilization_log = []
        self.queue_log = []
        self.idle_time = 0
        self.last_busy_time = 0
        self.env.process(self.breakdown_monitor()) # start breakdown monitoring
        self.env.process(self.log_time_series()) # start time series logging

    def monitor_utilization(self): #  track how much time a ride spends in each operational state
        now = self.env.now # gets current sim time
        duration = now - self.last_event_time # calc how much time passed since last state change
        state = 'broken' if self.broken else ('busy' if self.resource.count else 'idle')
        self.state_durations[state] += duration # add to state duration
        if state == 'idle':
            self.idle_time += duration # track idle time
        elif state == 'broken':
            self.last_busy_time = now # track last busy time
        self.last_event_time = now # track last event time

    def log_event(self, visitor_id, arrival_time, wait_time, service_time, ride_start_time, ride_end_time, visitor_type, priority):
        if len(event_logs_per_ride[self.name]) < 20:
            event_logs_per_ride[self.name].append([
                format_time(self.env.now),
                visitor_id,
                visitor_type,
                priority,
                self.name,
                format_time(arrival_time),
                f"{service_time} min",
                format_time(ride_start_time),
                format_time(ride_end_time),
                format_time(wait_time),
                len(self.resource.queue),
                self.resource.count,
                "broken" if self.broken else "busy" if self.resource.count else "idle"
            ])

    def breakdown_monitor(self): # sim random breakdowns
        while True:
            if not self.broken and random.random() < RIDE_FAILURE_PROB[self.name]: # chance of breakdown
                self.broken = True # mark as broken
                self.failures += 1
                self.monitor_utilization() # log state change
                yield self.env.timeout(RIDE_REPAIR_TIME()) # sim repair time
                self.monitor_utilization() # log repair finished
                self.broken = False # ride is fixed
            yield self.env.timeout(1) # check again in one minute

    def log_time_series(self): # log util and queue length every 5 mins
        while True:
            self.utilization_log.append((self.env.now, self.resource.count))
            self.queue_log.append((self.env.now, len(self.resource.queue)))
            yield self.env.timeout(5)

def visitor_generator(env, rides, arrival_mean):
    visitor_id = 0
    while True:
        yield env.timeout(np.random.exponential(arrival_mean)) # wait for next visitor
        visitor_type = random.choices(["child", "adult"], weights=[0.5, 0.5])[0] # visitor type
        preferred_rides = {
            "child": ["Carousel", "Slide Cars", "Self-Control Planes", "Spiral Rides"],
            "adult": ["Ferris Wheel", "Race Cars", "Flying Tower", "Spiral Rides"]
        }
        ride_name = random.choice(preferred_rides[visitor_type])
        ride = rides[ride_name] # calls ride object

        priority = 'VIP' if random.random() < 0.1 else 'regular' # 10% chance of vip
        env.process(visitor(env, visitor_id, ride, priority, visitor_type)) # start process
        visitor_id += 1

def visitor(env, visitor_id, ride, priority='regular', visitor_type='child'): # sim visitor behaviour
    arrival_time = env.now
    
    while ride.broken: # drop visitors if ride is broken
        ride.dropped_visitors += 1
        return
    
    with ride.resource.request(priority=VISITOR_PRIORITY[priority]) as request: # request access to ride
        yield request # wait
        ride.monitor_utilization() # log utilization state

        wait_time = env.now - arrival_time # calc wait time
        service_time = RIDE_SERVICE_TIME[ride.name]() # get service duration
        
        ride.total_visitors += 1
        ride.total_wait_time += wait_time
        ride.utilization_time += service_time
        ride.queue_lengths.append(len(ride.resource.queue))
        ride.completed_rides += 1

        ride_start_time = env.now
        yield env.timeout(service_time) # sim ride simulation
        ride_end_time = env.now

        ride.monitor_utilization() # update start again

        ride.log_event(visitor_id, arrival_time, wait_time, service_time, ride_start_time, ride_end_time, visitor_type, priority)

def visualize_summary(rides, sim_time):
    ride_names = list(rides.keys())
    
    utilization = []
    queue_lengths = []
    wait_times = []
    downtimes = []
    for ride in rides.values():
        max_util = ride.capacity * sim_time
        utilization.append((ride.utilization_time / max_util) * 100)
        avg_q_len = np.mean(ride.queue_lengths) if ride.queue_lengths else 0
        queue_lengths.append(avg_q_len)
        avg_wait = ride.total_wait_time / ride.total_visitors if ride.total_visitors else 0
        wait_times.append(avg_wait)
        downtimes.append(ride.state_durations['broken'] / 60)
    
    fig, axs = plt.subplots(2, 2, figsize=(14, 10))
    colors = plt.cm.Pastel1.colors
    
    axs[0,0].bar(ride_names, utilization, color=colors)
    axs[0,0].set_title("Ride Utilization", fontsize=10)
    axs[0,0].set_ylabel("Utilization (%)")
    axs[0,0].tick_params(axis='x', rotation=45)
    
    axs[0,1].bar(ride_names, wait_times, color=colors)
    axs[0,1].set_title("Average Wait Time", fontsize=10)
    axs[0,1].set_ylabel("Minutes")
    axs[0,1].tick_params(axis='x', rotation=45)
    
    axs[1,0].bar(ride_names, queue_lengths, color=colors)
    axs[1,0].set_title("Average Queue Length", fontsize=10)
    axs[1,0].set_ylabel("People")
    axs[1,0].tick_params(axis='x', rotation=45)
    
    axs[1,1].bar(ride_names, downtimes, color=colors)
    axs[1,1].set_title("Total Downtime", fontsize=10)
    axs[1,1].set_ylabel("Hours")
    axs[1,1].tick_params(axis='x', rotation=45)
    
    plt.suptitle("Theme Park Performance Summary", fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig("performance_summary.png")

def plot_time_series(rides, top_n=2):
    bottlenecks = []
    for name, ride in rides.items():
        avg_wait = ride.total_wait_time / ride.total_visitors if ride.total_visitors else 0
        util_pct = (ride.utilization_time / (ride.capacity * SIMULATION_TIME)) * 100
        bottlenecks.append((name, avg_wait, util_pct))
    bottlenecks.sort(key=lambda x: (-x[1], -x[2]))
    top_rides = [b[0] for b in bottlenecks[:top_n]]
    
    if not top_rides:
        return
    
    fig, axs = plt.subplots(len(top_rides), 2, figsize=(14, 5*len(top_rides)))
    if len(top_rides) == 1:
        axs = [axs]
    
    for i, ride_name in enumerate(top_rides):
        ride = rides[ride_name]
        times_u, util = zip(*ride.utilization_log)
        times_q, queue = zip(*ride.queue_log)

        axs[i][0].plot(times_u, util, color='#ff69b4')
        axs[i][0].set_title(f"{ride_name} - Utilization", fontsize=10)
        axs[i][0].set_ylabel("In Use")

        axs[i][1].plot(times_q, queue, color='#da70d6')
        axs[i][1].set_title(f"{ride_name} - Queue Length", fontsize=10)
        axs[i][1].set_ylabel("People")
    
    plt.suptitle("Time Series for Top Bottleneck Rides", y=1.02)
    plt.tight_layout()
    plt.savefig("bottleneck_rides_timeseries.png")

def plot_downtime_pie(rides):
    labels = []
    values = []
    for name, ride in rides.items():
        downtime = ride.state_durations['broken']
        if downtime > 0:
            labels.append(name)
            values.append(downtime)
    plt.figure(figsize=(7, 7))
    plt.pie(values, labels=labels, autopct='%1.1f%%', startangle=140,
            colors=plt.cm.Pastel1.colors)
    plt.title("Proportion of Downtime per Ride", color='hotpink')
    plt.savefig("downtime_pie.png")

def simulate():
    env = simpy.Environment()
    rides = {name: Ride(env, name, RIDE_CAPACITY[name]) for name in RIDE_NAMES}
    env.process(visitor_generator(env, rides, VISITOR_ARRIVAL_MEAN))
    env.run(until=SIMULATION_TIME)
    total_visitors = sum(r.total_visitors for r in rides.values())
    dropped_visitors = sum(r.dropped_visitors for r in rides.values())
    completed = sum(r.completed_rides for r in rides.values())
    avg_wait = sum(r.total_wait_time for r in rides.values()) / completed if completed else 0
    total = total_visitors + dropped_visitors

    print("\n♡ Logs ♡\n")
    combined_logs = []
    for logs in event_logs_per_ride.values():
        combined_logs.extend(logs)
    random.shuffle(combined_logs)
    headers = ["Minute", "Visitor ID", "Visitor Type", "Priority", "Ride", "Arrival Time", "Service Time",
               "Ride Start Time", "Ride End Time", "Wait Time", "Queue Length", "Riders", "Status"]

    print(tabulate(combined_logs[:50], headers=headers, tablefmt="fancy_grid"))

    print("\n♡ Simulation Summary ♡")
    print(f"Simulation Time: {SIMULATION_TIME} minutes")
    print(f"Total Visitors: {total}")
    print(f"Dropped Visitors: {dropped_visitors}")
    print(f"Completed Visitors: {completed}")
    print(f"Average Wait Time: {format_time(avg_wait)}")

    bottlenecks = [] # list to stpre bottleneck data for each ride
    for name, ride in rides.items():
        util_pct = (ride.utilization_time / (ride.capacity * SIMULATION_TIME)) * 100 # calc util %
        avg_q = np.mean(ride.queue_lengths) if ride.queue_lengths else 0
        avg_wait = ride.total_wait_time / ride.total_visitors if ride.total_visitors else 0
        down_time = ride.state_durations['broken']
        down_pct = (down_time / SIMULATION_TIME) * 100
        bottlenecks.append((name, avg_wait, util_pct)) # store name, avg wait, util for analysis

        print(f"\n♡ {name} ♡")
        print(f"  - Capacity: {ride.capacity}")
        print(f"  - Served: {ride.total_visitors}")
        print(f"  - Dropped Visitors: {ride.dropped_visitors}")
        print(f"  - Failures: {ride.failures}")
        print(f"  - Utilization: {util_pct:.2f}%")
        print(f"  - Downtime: {format_time(down_time)} ({down_pct:.2f}%)")
        print(f"  - Avg Queue: {avg_q:.2f}")
        print(f"  - Avg Wait: {format_time(avg_wait)}")

    bottlenecks.sort(key=lambda x: (-x[1], -x[2]))

    print("\n♡ Top Bottleneck Rides ♡")
    for b in bottlenecks[:2]:
        print(f"{b[0]} - Avg Wait: {b[1]:.2f} min, Utilization: {b[2]:.2f}%")
    visualize_summary(rides, SIMULATION_TIME)
    plot_time_series(rides)
    plot_downtime_pie(rides)

simulate()