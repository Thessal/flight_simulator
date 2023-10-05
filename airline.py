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
        "std_delay":5,
        "late_threshold":15,
        "holding_period":120,
        "timestep":5,
        "capacity":10,
        "num_plane":100,
        "num_iters":30000,
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
        fight_time = self.df_time.loc[(origin,destination)]
        return {
            "org":origin, "dst":destination, 
            "time_plan":self.time+fight_time, 
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
    
    def __init__(self, df_preference, code, std_delay=5, late_threshold=15, holding_period=120, timestep=5, capacity=10, debug=False):
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
        # self.df_inventory = pd.DataFrame({"id":[], "dst":[], "time_plan":[], "time_est":[]})
        self.arrivals = {}
        self.debug=debug
        self.policy_lst = POLICY
        self.capacity = capacity
        self.policy = DEFAULT_POLICY
        
        # consider delay accumulation of Korean airport only
        self.accumulate_delay = True if code.startswith("RK") else False
    
    def calc_delay(self):
        return np.random.normal(0,self.std_delay)
    
    def update(self, df_arrivals):
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
            if num_landed + len(arrivals) > self.capacity:
                break
            arrivals[self.plane_id] = {
                "landing_plan":info["time_plan"],
                "landing_time":self.time,
                "takeoff_plan":info["time_plan"] + self.holding_period,
                "takeoff_ready": (
                    self.time + self.holding_period + self.calc_delay() if self.accumulate_delay
                    else info["time_plan"] + self.holding_period + self.calc_delay()
                ),
                }
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
        next_station = np.random.choice(self.p.index, p=self.p.values)
        delay = max(0,self.time-flight["takeoff_plan"])
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
    def __init__(self, cfg, dfs=None, add_flights=True):
        self.cfg = cfg
        self.add_flights = add_flights 
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
            std_delay = self.cfg["std_delay"], 
            late_threshold = self.cfg["late_threshold"], 
            holding_period = self.cfg["holding_period"], 
            timestep = self.cfg["timestep"], 
            capacity = self.cfg["capacity"], 
            debug = self.cfg["debug"]
            ) for icao in self.df_airline.values}

    def step(self): 
        arrivals = self.sky.arrivals
        flight_info_lst = []
        arrival_ids = []
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
        # TODO
        # reward = [x for x in reward if (x[0].startswith("RK") and x[1].startswith("RK"))]
        return reward
    
if __name__ =="__main__":
    ## Test
    
    dfs = get_df()
    CONFIG.update({"timestep":5, "capacity":3, "debug":True})
    sim = Simulator(CONFIG, dfs, add_flights=False)
    sim.sky.update(
        [
        ("RKPC","RKSI",30),("RKPC","RKSI",10),
        ("RKPC","RKSI",20),("RKPC","RKSI",20),("RKPC","RKSI",20),("RKPC","RKSI",20)
        ],[])

    actions = {x:DEFAULT_POLICY for x in sim.airports}
    for i in range(60):
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