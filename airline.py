import numpy as np 
import scipy.stats as ss
import pandas as pd
import geopy.distance

# def black_scholes(S,K,ttm,sigma):
#     # S : current
#     # K : strike
#     sigma_tau = sigma * (ttm**0.5)
#     d1 = ( np.log(S/K) + (0.5*(sigma**2)) * ttm ) / ( sigma_tau ) if S>0 else -np.inf
#     d2 = d1 - sigma_tau
#     return ss.norm.cdf(d1) * S - ss.norm.cdf(d2) * K
#     #pd.DataFrame({ttm:{delay: bs(delay,15,ttm,0.1) for delay in np.arange(0,30,0.1)} for ttm in (1,10,20,30)}).plot()

CONFIG = {
        "std_delay":10,
        "buffer_time":60, # 30 
        "late_threshold":15, 
        "holding_period":15,
        "timestep":10,
        "capacity":10,
        "num_plane":100,
        "num_iters":30000,
        "agent_airports":["RKSI", "RKSS", "RKPK", "RKPC", "RKTN", "RKTU", "RKJB"], # Only agent_airports are optimized, other airports acts as dummy (immediately removes delay)
        "return_p":0.0,
        "debug":False,
        }
DEFAULT_POLICY = np.int64(0)
POLICY_NAMES = [
    "FIFO",
    "Urgent first",
    "Maximize buffer"
]
POLICY = [
            lambda costs: min(costs,key=lambda x:costs[x]["landing_time"]), # FIFO
            lambda costs: max(costs,key=lambda x:costs[x]["intrinsic_cost"]), # urgent first, based on arrival time
            lambda costs: max(costs,key=lambda x:costs[x]["est_cost"] - costs[x]["intrinsic_cost"]), # urgent first, based on profit
        ]

class Sky:
    def __init__(self, df_time, timestep = 5, debug=False):
        self.df_flight = pd.DataFrame({"org":[], "dst":[], "time_plan":[], "delay":[]})
        self.df_time = df_time 
#         self.balance = 0
        self.time = 0
        self.timestep = timestep
        self.debug = debug

    def update(self, flight_info, landed_idx):
        self.df_flight = self.df_flight[~self.df_flight.index.isin(landed_idx)]
        new_flights = pd.DataFrame([self._new_flight(info) for info in flight_info])
        self.df_flight = pd.concat(
            [self.df_flight, new_flights]
        ).sort_values("time_plan").reset_index(drop=True)
        if self.debug and (len(new_flights)>0):
            print(new_flights)
        self.time += self.timestep
        return new_flights
        
    def _new_flight(self, flight_info):
        origin, destination, delay = flight_info
        flight_time = self.df_time.loc[(origin,destination)]
        return {
            "org":origin, "dst":destination, 
            "time_plan":self.time-delay+flight_time,
            # "time_plan":self.time+flight_time, 
            "delay":delay}
    
    def add_random_flight(self, df_preference, n):
        tmp = df_preference.stack()
        pairs = np.random.choice(tmp.index, n, replace=False, p=tmp.values/tmp.sum())
        self.update([(orig, dest, 0) for orig, dest in pairs], [])
        
    @property
    def arrivals(self):
        arrivals = self.df_flight[(self.df_flight["time_plan"]+self.df_flight["delay"])<=self.time]
        return dict(list(arrivals.groupby("dst")))
    

class Airport:
    def observe(self):
        # delays = [ min(60,max(-60, self.time - x["takeoff_plan"])) for x in self.arrivals.values() ]
        # delays = sorted(delays, reverse=True)
        delays = [ (x["landing_time"] - x["landing_plan"]) for x in self.landed.values()]
        return (
            np.array([min(len(delays),10)], dtype=np.float32), 
            np.array([max(min(sum(delays),60),-60)], dtype=np.float32)
            )
    
    def __init__(self, df_preference, code, accumulate_delay=True, std_delay=5, buffer_time=5, late_threshold=15, holding_period=120, timestep=5, capacity=10, return_p=0.2, debug=False):
        self.late_threshold = 15
        self.holding_period = holding_period # minutes between landing and takeoff. (~TTM)
        self.code = code
        self.time = 0
        self.timestep = timestep
        self.plane_id = 0
        pref = df_preference.loc[code]
        self.p = pref / (pref.sum())
        self.sigma = std_delay / late_threshold
        self.std_delay = std_delay
        self.buffer_time = buffer_time
        # self.df_inventory = pd.DataFrame({"id":[], "dst":[], "time_plan":[], "time_est":[]})
        self.arrivals = {}
        self.debug=debug
        self.policy_lst = POLICY
        self.capacity = capacity
        self.policy = DEFAULT_POLICY
        self.return_p = return_p
        
        # If accumulate_delay is false, landing delay is ignored and immediately takeoff
        self.accumulate_delay = accumulate_delay
    
    def calc_delay(self):
        return np.random.normal(0,self.std_delay)
    
    def update(self, df_arrivals):
        # if self.code == "RKSI" and self.time>2000:
        #     print(df_arrivals)
        #     print("debug")
        if df_arrivals is not None:
            arrival_ids = self._land(df_arrivals)
        else:
            arrival_ids = []
        costs = {code: self._est_cost(code) for code,info in self.arrivals.items() if info["takeoff_ready"]<=self.time}
        if self.debug and len(costs)>0:
            print(f"{self.code}[{self.time}] Costs : \n{costs}")
        plane_id = self._select_plane(costs)
        if plane_id is not None:
            flight_info = self._takeoff(plane_id) #( self.code, next_station, delay )
            cost = flight_info[2]
        else:
            flight_info = None
            cost = None
        # print(cost) # TODO
        self.time += self.timestep
        return flight_info, arrival_ids
        
    def _select_plane(self, costs):
        if len(costs)>0:
            return self.policy_lst[self.policy](costs)
        else:
            return None
    
    @property
    def landed(self):
        return {idx:plane for idx, plane in self.arrivals.items() if (plane["landing_time"]<=self.time)}

    def _land(self, df_arrivals):
        arrivals = dict()
        arrival_ids = []
        for idx, info in df_arrivals.iterrows():
            num_landed = len(self.landed)
            landed_time = info["time_plan"]+info["delay"] # assuming no delay during flight
            if num_landed + len(arrivals) > self.capacity:
                break
            elif landed_time > self.time : 
                continue
            landed_time = max(landed_time, self.time-self.timestep)

            takeoff_plan = info["time_plan"] + self.holding_period
            arrivals[self.plane_id] = {
                "origin":info["org"],
                "landing_plan":info["time_plan"],
                "landing_time":landed_time,
                "takeoff_plan":takeoff_plan,
                "takeoff_ready": (
                    max(
                        takeoff_plan, 
                        landed_time + self.holding_period - self.buffer_time + self.calc_delay()
                    )
                    # max(self.time + self.holding_period - self.buffer_time + self.calc_delay(), info["time_plan"] + self.holding_period)
                    if self.accumulate_delay
                    else info["time_plan"] + self.holding_period
                ),
                }
            # if self.code=="RKSI" and info["delay"]>10:
            #     print(info["delay"])
            self.plane_id += 1
            arrival_ids.append(idx)
        if self.debug and len(arrivals)>0:
            debugstr = '\n'.join([str((k,v)) for k,v in arrivals.items()])
            print(f"{self.code}[{self.time}] Landing : \n{debugstr}")
        self.arrivals.update(arrivals)
        return arrival_ids
        
    def _est_cost(self, plane_id):
        x = self.arrivals[plane_id]
        orig_delay = x["landing_time"] - x["landing_plan"]
        intrinsic_cost = max(0,orig_delay-self.late_threshold)
        est_cost = self.black_scholes(orig_delay, self.late_threshold, 1, self.sigma)
        return {"landing_time": x["landing_time"], "intrinsic_cost":intrinsic_cost, "est_cost":est_cost}
        
    def _takeoff(self, plane_id):
        flight = self.arrivals.pop(plane_id)
        assert(flight["takeoff_ready"]<=self.time)
        if np.random.random()<self.return_p:
            next_station = flight["origin"] # Return to where it came from
        else:
            next_station = np.random.choice(self.p.index, p=self.p.values)
        delay = max(-30,flight["takeoff_ready"]-flight["takeoff_plan"])
        info = ( self.code, next_station, delay )
        if self.debug:
            print(f"{self.code} Takeoff : plane_id: {plane_id}, info: {info}")
        return info
            
    @staticmethod
    def black_scholes(S,K,ttm,sigma):
        # S : current
        # K : strike
        sigma_tau = sigma * (ttm**0.5)
        d1 = ( np.log(S/K) + (0.5*(sigma**2)) * ttm ) / ( sigma_tau ) if S>0 else -np.inf
        d2 = d1 - sigma_tau
        return ss.norm.cdf(d1) * S - ss.norm.cdf(d2) * K
        #pd.DataFrame({ttm:{delay: bs(delay,15,ttm,0.1) for delay in np.arange(0,30,0.1)} for ttm in (1,10,20,30)}).plot()

def get_df():
    import os.path
    path = "/home/jongkook90/riskds/rl/"
    if os.path.isfile(path+"df_preference.pkl") :
        df_preference = pd.read_pickle(path+"df_preference.pkl")
        df_airports = pd.read_pickle(path+"df_airports.pkl")
    else:
        df = pd.read_pickle("analysis.pkl")
        df_preference_stack = df.groupby(["출발지","도착지"])["예상"].count()
        df_preference = df_preference_stack.unstack().fillna(0)
        df_loc = pd.read_csv("./kaggle/airports.csv")
        airports = list(set(df_preference.columns).intersection(set(df_preference.index)).intersection(set(df_loc["ICAO"].values)))
        df_preference = df_preference.loc[airports,airports]
        df_airports = pd.Series(airports)
        df_preference.to_pickle("df_preference.pkl")
        df_airports.to_pickle("df_airports.pkl")

    if os.path.isfile(path+"df_time.pkl"):
        df_time = pd.read_pickle(path+"df_time.pkl")
    else:
        df_loc = pd.read_csv("./kaggle/airports.csv")
        df_loc = df_loc.set_index("ICAO")[["Latitude","Longitude"]]
        df_loc = df_loc.loc[list(set(df_loc.index).intersection(set(df_preference.columns).union(df_preference.index)))]
        infer_time = lambda lat1, lon1, lat2, lon2 : geopy.distance.geodesic((lat1, lon1), (lat2,lon2)).km * 0.08 #(~750km/h)

        result = dict()
        for i1,x1 in df_loc.iterrows():
            buf = dict()
            for i2,x2 in df_loc.iterrows():
                buf[i2] = infer_time(x1["Latitude"],x1["Longitude"],x2["Latitude"],x2["Longitude"])
            result[i1] = buf
        df_time = pd.DataFrame(result).stack()
        df_time.to_pickle("df_time.pkl")
    
    return df_airports, df_preference, df_time
    
class Simulator:
    def __init__(self, cfg, dfs=None, add_flights=True, save_history=False):
        self.cfg = cfg
        self.add_flights = add_flights 
        self.history = {
            "current":
            {
                k : {icao:0 for icao in self.cfg["agent_airports"]}
                for k in ["incoming_count","incoming_delay", "outgoing_count", "outgoing_delay"]
            }
        }
        self.save_history = save_history
        if dfs is None:
            self.df_airline, self.df_preference, self.df_time = get_df()
        else:
            self.df_airline, self.df_preference, self.df_time = dfs
        self.reset()

    def reset(self):
        self.reset_sky()
        self.reset_airports()
        if self.add_flights:
            self.sky.add_random_flight(self.df_preference, n=self.cfg["num_plane"])

    def reset_sky(self):
        self.sky = Sky(self.df_time, timestep = self.cfg["timestep"])

    def reset_airports(self):
        self.airports = {icao:Airport( 
            self.df_preference, icao, 
            accumulate_delay = True, #icao in self.cfg["agent_airports"],
            std_delay = self.cfg["std_delay"], 
            buffer_time = self.cfg["buffer_time"],
            late_threshold = self.cfg["late_threshold"], 
            holding_period = self.cfg["holding_period"], 
            timestep = self.cfg["timestep"], 
            capacity = self.cfg["capacity"], 
            return_p = self.cfg["return_p"], 
            debug = self.cfg["debug"]
            ) for icao in self.df_airline.values}

    def step(self): 
        arrivals = self.sky.arrivals
        flight_info_lst = []
        arrival_ids = []
        if self.save_history:
            df_arrivals = pd.concat(arrivals.values()).copy() if not all((x is None) for x in arrivals.values()) else None
        for code, airport in self.airports.items():
            if code in arrivals:
                flight_info, landed_ids = airport.update( arrivals[code] )
            else:
                flight_info, landed_ids = airport.update( None )
            if flight_info is not None:
                flight_info_lst.append(flight_info)#( self.code, next_station, delay )
            arrival_ids.extend(landed_ids)
        if self.add_flights:
            new_flights = self.sky.update(flight_info_lst, arrival_ids)
        else: # debug
            new_flights = self.sky.update([], arrival_ids)

        # pd.DataFrame({"org":[], "dst":[], "time_plan":[], "delay":[]})
        cost = lambda x : max(x-15, 0)
        reward = [(x['org'], x['dst'], -cost(x['delay']), cost(x['delay'])) for _,x in new_flights.iterrows()]
        # TODO : modify cost, reward
        # reward = [x for x in reward if (x[0] in self.cfg["agent_airports"]) and (x[1] in self.cfg["agent_airports"])] ## consider only interaction between korean airports
        if self.save_history:
            arrivals = df_arrivals.loc[arrival_ids] if (df_arrivals is not None) else None
            departures = new_flights.copy()
            if (arrivals is not None) and len(arrivals)>0 :
                for icao,x in arrivals.groupby("dst"):
                    if icao in self.history["current"]["incoming_delay"]:
                        self.history["current"]["incoming_delay"][icao] += x["delay"].sum()
                        self.history["current"]["incoming_count"][icao] += len(x)
            if (departures is not None) and len(departures)>0 :
                for icao,x in departures.groupby("org"):
                    if icao in self.history["current"]["outgoing_delay"]:
                        self.history["current"]["outgoing_delay"][icao] += x["delay"].sum()
                        self.history["current"]["outgoing_count"][icao] += len(x)
            self.history[self.sky.time] = {
                "num_flights":len(self.sky.df_flight),
                "arrivals": arrivals,
                "departures": departures,
                "policy":{icao:self.airports[icao].policy.copy() for icao in self.cfg["agent_airports"]},
                "landed":{icao:self.airports[icao].landed.copy() for icao in self.cfg["agent_airports"]},
                "incoming_delay": self.history["current"]["incoming_delay"].copy(),
                "incoming_count": self.history["current"]["incoming_count"].copy(),
                "outgoing_delay": self.history["current"]["outgoing_delay"].copy(),
                "outgoing_count": self.history["current"]["outgoing_count"].copy(),
                "reward": reward,
                }
        return reward
    
if __name__ =="__main__":
    ## Test
    
    dfs = get_df()
    # config = CONFIG.copy()
    # config.update({"timestep":5, "capacity":3, "debug":True})
    # sim = Simulator(config, dfs, add_flights=False)
    # sim.sky.update(
    #     [
    #     ("RKPC","RKSI",30),("RKPC","RKSI",10),
    #     ("RKPC","RKSI",20),("RKPC","RKSI",20),("RKPC","RKSI",20),("RKPC","RKSI",20)
    #     ],[])

    config = {'std_delay': 10, 'buffer_time': 5, 'late_threshold': 15, 'holding_period': 60, 'timestep': 10, 'capacity': 5, 'num_plane': 100, 'num_iters': 30000, 'agent_airports': ['RKSI', 'RKSS', 'RKPK', 'RKPC', 'RKTN', 'RKTU', 'RKJB'], 'return_p': 0, 'debug': False}
    sim = Simulator(config, dfs, add_flights=True)

    actions = {x:DEFAULT_POLICY for x in sim.airports}
    for i in range(600):
        sim.step()
    # airports, df_preference, df_time = get_df()
    # sky = Sky(df_time)
    # airport_RKSI = Airport( df_preference, "RKSI", debug=True)
    # airports = {"RKSI":airport_RKSI}

    # sky.update([

    # for i in range(60):
    #     arrivals = sky.arrivals
    #     # print(arrivals)
    #     # print(sky.df_flight)
    #     arrival_ids = []
    #     for code, airport in airports.items():
    #         if code in arrivals:
    #             flight_info, landed_ids = airport.update( arrivals[code] )
    #         else:
    #             flight_info, landed_ids = airport.update( None )
    #         arrival_ids.extend(landed_ids)
    #     ## sky.update(flight_info_lst, [i for v in arrivals.values() for i in v.index])
    #     print(arrival_ids)
    #     sky.update([], arrival_ids)