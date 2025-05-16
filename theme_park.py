import random
import numpy as np
import matplotlib.pyplot as plt
from collections import deque, defaultdict
from tabulate import tabulate

SIMULATION_TIME = 8 * 60
VISITOR_ARRIVAL_MEAN = 2
RIDE_NAMES = ["Carousel", "Slide Cars", "Race Cars", "Ferris Wheel",
              "Self-Control Planes", "Spiral Rides", "Flying Tower"]
RIDE_CAPACITY = {
    "Carousel": 10,
    "Slide Cars": 8,
    "Race Cars": 5,
    "Ferris Wheel": 20,
    "Self-Control Planes": 6,
    "Spiral Rides": 7,
    "Flying Tower": 10,
}
RIDE_SERVICE_TIME = {
    "Carousel": lambda: int(np.random.normal(5, 1)),
    "Slide Cars": lambda: int(np.random.triangular(3, 5, 8)),
    "Race Cars": lambda: int(np.random.normal(7, 2)),
    "Ferris Wheel": lambda: int(np.random.normal(10, 3)),
    "Self-Control Planes": lambda: int(np.random.triangular(4, 6, 9)),
    "Spiral Rides": lambda: int(np.random.normal(6, 1)),
    "Flying Tower": lambda: int(np.random.triangular(5, 8, 12)),
}
RIDE_FAILURE_PROB = 0.01
RIDE_REPAIR_TIME = lambda: int(np.random.weibull(1.5) * 10)

event_logs_per_ride = {name: [] for name in RIDE_NAMES}

class Ride:
    def __init__(self, name, capacity):
        self.name = name
        self.capacity = capacity
        self.queue = deque()
        self.active_riders = []
        self.status = "idle"
        self.repair_time_remaining = 0
        self.utilization = 0
        self.completed_rides = 0
        self.total_queue_length = 0
        self.total_wait_time = 0
        self.total_visitors = 0
        self.state_durations = {
            'idle': 0,
            'busy': 0,
            'broken': 0
        }

    def update(self, minute):
        if self.status == "broken":
            self.repair_time_remaining -= 1
            if self.repair_time_remaining <= 0:
                self.status = "idle"
        else:
            if random.random() < RIDE_FAILURE_PROB:
                self.status = "broken"
                self.repair_time_remaining = RIDE_REPAIR_TIME()
                self.active_riders.clear()
            else:
                for rider in list(self.active_riders):
                    rider['remaining'] -= 1
                    if rider['remaining'] <= 0:
                        self.active_riders.remove(rider)

                while len(self.active_riders) < self.capacity and self.queue:
                    visitor = self.queue.popleft()
                    wait_time = minute - visitor['arrival']
                    service_time = RIDE_SERVICE_TIME[self.name]()
                    visitor['waited'] = wait_time
                    self.total_wait_time += wait_time
                    self.total_visitors += 1
                    self.active_riders.append({
                        'id': visitor['id'],
                        'remaining': service_time
                    })
                    self.completed_rides += 1

                    if len(event_logs_per_ride[self.name]) < 20:
                        event_logs_per_ride[self.name].append([
                            minute,
                            visitor['id'],
                            self.name,
                            visitor['arrival'],
                            service_time,
                            wait_time,
                            len(self.queue),
                            len(self.active_riders),
                            self.status
                        ])

        if self.status != "broken":
            self.status = "busy" if self.active_riders else "idle"

        self.state_durations[self.status] += 1
        self.utilization += len(self.active_riders)
        self.total_queue_length += len(self.queue)

def generate_arrivals(max_time):
    arrivals = []
    time = 0
    id_counter = 0
    while time < max_time:
        interarrival = int(np.random.exponential(VISITOR_ARRIVAL_MEAN))
        time += interarrival
        if time >= max_time:
            break
        ride = random.choice(RIDE_NAMES)
        arrivals.append({'id': id_counter, 'arrival': time, 'ride': ride})
        id_counter += 1
    return arrivals

def visualize_summary(rides, sim_time, completed_visitors, total_visitors):
    ride_names = list(rides.keys())
    utilization = []
    queue_lengths = []
    wait_times = []

    for ride in rides.values():
        max_util = ride.capacity * sim_time
        utilization.append((ride.utilization / max_util) * 100)
        queue_lengths.append(ride.total_queue_length / sim_time)
        avg_wait = (ride.total_wait_time / ride.total_visitors) if ride.total_visitors else 0
        wait_times.append(avg_wait)

    throughput_percent = (completed_visitors / total_visitors) * 100

    # Soft pastel theme
    pastel_pink = "#ffb6c1"
    pastel_purple = "#dda0dd"
    pastel_peach = "#ffe4e1"

    plt.rcParams.update({
        "axes.facecolor": "#fff0f5",
        "figure.facecolor": "#fff0f5",
        "axes.titleweight": "bold",
        "axes.titlesize": 14,
        "axes.labelsize": 12
    })

    fig, axs = plt.subplots(3, 1, figsize=(12, 10))

    axs[0].bar(ride_names, utilization, color=pastel_pink, edgecolor='hotpink')
    axs[0].set_title("Ride Utilization")
    axs[0].set_ylabel("Utilization (%)")

    axs[1].bar(ride_names, queue_lengths, color=pastel_purple, edgecolor='mediumorchid')
    axs[1].set_title("Average Queue Length")
    axs[1].set_ylabel("People/Minute")

    axs[2].bar(ride_names, wait_times, color=pastel_peach, edgecolor='salmon')
    axs[2].set_title("Average Wait Time")
    axs[2].set_ylabel("Minutes")
    axs[2].set_xlabel("Ride Name")

    for ax in axs:
        ax.tick_params(axis='x', rotation=15)
        ax.spines[['top', 'right']].set_visible(False)

    plt.suptitle(f"Theme Park Summary (Throughput: {throughput_percent:.2f}%)", fontsize=18, color='deeppink', weight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig("plot.png")

def simulate(sim_time=SIMULATION_TIME, arrival_mean=VISITOR_ARRIVAL_MEAN):
    rides = {name: Ride(name, RIDE_CAPACITY[name]) for name in RIDE_NAMES}
    arrivals = generate_arrivals(sim_time)
    timeline = defaultdict(list)
    for visitor in arrivals:
        timeline[visitor['arrival']].append(visitor)

    total_failures = {name: 0 for name in RIDE_NAMES}
    served_ids = set()

    for minute in range(sim_time):
        for visitor in timeline[minute]:
            rides[visitor['ride']].queue.append(visitor)

        for name, ride in rides.items():
            prev_status = ride.status
            ride.update(minute)
            if prev_status != "broken" and ride.status == "broken":
                total_failures[name] += 1

            served_ids.update(r['id'] for r in ride.active_riders)

    print("\n--- Simulation Summary ---")
    print(f"Simulation Time: {sim_time} minutes")

    total_visitors = len(arrivals)
    completed_visitors = sum(ride.completed_rides for ride in rides.values())
    dropped_visitors = total_visitors - completed_visitors
    dropped_ids = sorted(set(v['id'] for v in arrivals) - served_ids)
    total_wait = sum(ride.total_wait_time for ride in rides.values())
    avg_wait = total_wait / completed_visitors if completed_visitors else 0

    print(f"Total Visitors: {total_visitors}")
    print(f"Completed Visitors: {completed_visitors}")
    print(f"Dropped Visitors: {dropped_visitors}")
    print(f"Average Wait Time: {avg_wait:.2f} minutes")

    bottlenecks = []
    for name, ride in rides.items():
        max_possible_utilization = ride.capacity * sim_time
        utilization_percent = (ride.utilization / max_possible_utilization) * 100
        avg_queue_length = ride.total_queue_length / sim_time
        avg_wait_time = ride.total_wait_time / ride.total_visitors if ride.total_visitors else 0
        downtime = ride.state_durations['broken']
        downtime_percent = (downtime / sim_time) * 100

        bottlenecks.append((name, avg_wait_time, utilization_percent))

        print(f"\n{name}:")
        print(f"  Capacity: {ride.capacity}")
        print(f"  Visitors Served: {ride.total_visitors}")
        print(f"  Utilization: {utilization_percent:.2f}%")
        print(f"  Failures: {total_failures[name]}")
        print(f"  Downtime: {downtime} min ({downtime_percent:.2f}%)")
        print(f"  Avg Queue Length: {avg_queue_length:.2f} people/min")
        print(f"  Avg Wait Time: {avg_wait_time:.2f} min")

    bottlenecks.sort(key=lambda x: (-x[1], -x[2]))
    print("\n--- Top Bottleneck Rides (By Avg Wait & Utilization) ---")
    for b in bottlenecks[:2]:
        print(f"{b[0]} - Avg Wait: {b[1]:.2f} min, Utilization: {b[2]:.2f}%")

    print("\n--- Combined Per-Ride Simulation Table (Sorted by Minute) ---\n")
    combined_logs = []
    for logs in event_logs_per_ride.values():
        combined_logs.extend(logs)
    combined_logs.sort(key=lambda x: x[1])
    headers = ["Minute", "Visitor ID", "Ride", "Arrival Time", "Service Time",
               "Wait Time", "Queue Length", "Riders on Ride", "Ride Status"]
    print(tabulate(combined_logs, headers=headers, tablefmt="github"))

    print(f"\n--- Dropped Visitor IDs (not served): ---\n{dropped_ids}")

    visualize_summary(rides, sim_time, completed_visitors, len(arrivals))

simulate()