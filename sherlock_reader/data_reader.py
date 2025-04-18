#===============================================================================================
# Project: Predicting Commercial Flight Trajectories Using Transformers for CS 555
# Author(s): Luke, Alex
# Description: Classes responsible for reading and representing flight record data
#===============================================================================================

from .csv_tools import CSVReader, peek


class Header:
    """
    Represents a Type 2 record: general flight metadata and identifiers.
    """

    def __init__(self,
        rec_time: float,
        flt_key: int,
        bcn_code: int,
        cid: str,
        source: str,
        msg_type: str,
        acid: str,
        rec_type_cat: int,
        ac_type: str,
        origin: str,
        dest: str,
        ops_type: str,
        est_origin: str,
        est_dest: str,
        mode_s_code: str
        ) -> None:
        self.rec_time = rec_time
        self.flt_key = flt_key
        self.bcn_code = bcn_code
        self.cid = cid
        self.source = source
        self.msg_type = msg_type
        self.acid = acid
        self.rec_type_cat = rec_type_cat
        self.ac_type = ac_type
        self.origin = origin
        self.dest = dest
        self.ops_type = ops_type
        self.est_origin = est_origin
        self.est_dest = est_dest
        self.mode_s_code = mode_s_code

    @staticmethod
    def from_line(line: list[str]):
        separated = iter(line)

        # Record type, we know this is 3 already
        _ = next(separated)
        rec_time = float(next(separated))
        flt_key = int(next(separated))
        bcn_code_ = next(separated)
        bcn_code = None if not bcn_code_ else int(bcn_code_)
        cid = next(separated)
        source = next(separated)
        msg_type = next(separated)
        acid = next(separated)
        rec_type_cat = int(next(separated))
        ac_type = next(separated)
        origin = next(separated)
        dest = next(separated)
        ops_type = next(separated)
        est_origin = next(separated)
        est_dest = next(separated)
        mode_s_code = next(separated)

        header = Header(
            rec_time,
            flt_key,
            bcn_code,
            cid,
            source,
            msg_type,
            acid,
            rec_type_cat,
            ac_type,
            origin,
            dest,
            ops_type,
            est_origin,
            est_dest,
            mode_s_code
        )

        return header


class FlightPlan:
    """
    Represents a Type 4 record: full flight plan description including routing and altitudes.
    """

    def __init__(self,
        rec_time: float,
        flt_key: int,
        bcn_code: int,
        cid: str,
        source: str,
        msg_type: str,
        acid: str,
        rec_type_cat: int,
        ac_type: str,
        origin: str,
        dest: str,
        alt_code: str,
        alt: float,
        max_alt: float,
        assigned_alt_string: str,
        requested_alt_string: str,
        route: str,
        est_time: str,
        flt_cat: str,
        perf_cat: str,
        ops_type: str,
        equip_list: str,
        coordination_time: float,
        coordination_time_type: str,
        leader_dir: int,
        scratch_pad_1: str,
        scratch_pad_2: str,
        fix_pair_scratch_pad: str,
        pref_dep_arr_route: str,
        pref_dep_route: str,
        pref_arr_route: str,
        coordination_point: str,
        coordination_point_type: str,
        track_number: str,
        mode_s_code: str
        ) -> None:
        self.rec_time = rec_time
        self.flt_key = flt_key
        self.bcn_code = bcn_code
        self.cid = cid
        self.source = source
        self.msg_type = msg_type
        self.acid = acid
        self.rec_type_cat = rec_type_cat
        self.ac_type = ac_type
        self.origin = origin
        self.dest = dest
        self.alt_code = alt_code
        self.alt = alt
        self.max_alt = max_alt
        self.assigned_alt_string = assigned_alt_string
        self.requested_alt_string = requested_alt_string
        self.route = route
        self.est_time = est_time
        self.flt_cat = flt_cat
        self.perf_cat = perf_cat
        self.ops_type = ops_type
        self.equip_list = equip_list
        self.coordination_time = coordination_time
        self.coordination_time_type = coordination_time_type
        self.leader_dir = leader_dir
        self.scratch_pad_1 = scratch_pad_1
        self.scratch_pad_2 = scratch_pad_2
        self.fix_pair_scratch_pad = fix_pair_scratch_pad
        self.pref_dep_arr_route = pref_dep_arr_route
        self.pref_dep_route = pref_dep_route
        self.pref_arr_route = pref_arr_route
        self.coordination_point = coordination_point
        self.coordination_point_type = coordination_point_type
        self.track_number = track_number
        self.mode_s_code = mode_s_code

    @staticmethod
    def from_line(line: list[str]):
        """
        Parses a Type 4 flight plan line from the CSV.

        Args:
            line (list[str]): Tokenized CSV line.

        Returns:
            FlightPlan: The parsed flight plan entry.
        """

        separated = iter(line)

        # Record type, we know this is 3 already
        _ = next(separated)
        rec_time = float(next(separated))
        flt_key = int(next(separated))
        bcn_code_ = next(separated)
        bcn_code = None if not bcn_code_ else int(bcn_code_)
        cid = next(separated)
        source = next(separated)
        msg_type = next(separated)
        acid = next(separated)
        rec_type_cat = int(next(separated))
        ac_type = next(separated)
        origin = next(separated)
        dest = next(separated)
        alt_code = next(separated)
        alt_ = next(separated)
        alt = None if not alt_ else float(alt_)
        max_alt_ = next(separated)
        max_alt = None if not max_alt_ else float(max_alt_)
        assigned_alt_string = next(separated)
        requested_alt_string = next(separated)
        route = next(separated)
        est_time = next(separated)
        flt_cat = next(separated)
        perf_cat = next(separated)
        ops_type = next(separated)
        equip_list = next(separated)
        coordination_time_ = next(separated)
        coordination_time = None if not coordination_time_ else float(coordination_time_)
        coordination_time_type = next(separated)
        leader_dir_ = next(separated)
        leader_dir = None if not leader_dir_ else int(leader_dir_)
        scratch_pad_1 = next(separated)
        scratch_pad_2 = next(separated)
        fix_pair_scratch_pad = next(separated)
        pref_dep_arr_route = next(separated)
        pref_dep_route = next(separated)
        pref_arr_route = next(separated)
        coordination_point = next(separated)
        coordination_point_type = next(separated)
        track_number = next(separated)
        mode_s_code = next(separated)

        flight_plan = FlightPlan(
            rec_time,
            flt_key,
            bcn_code,
            cid,
            source,
            msg_type,
            acid,
            rec_type_cat,
            ac_type,
            origin,
            dest,
            alt_code,
            alt,
            max_alt,
            assigned_alt_string,
            requested_alt_string,
            route,
            est_time,
            flt_cat,
            perf_cat,
            ops_type,
            equip_list,
            coordination_time,
            coordination_time_type,
            leader_dir,
            scratch_pad_1,
            scratch_pad_2,
            fix_pair_scratch_pad,
            pref_dep_arr_route,
            pref_dep_route,
            pref_arr_route,
            coordination_point,
            coordination_point_type,
            track_number,
            mode_s_code
        )

        return flight_plan


class TrackPoint:
    """
    Represents a Type 3 record: positional fix of the aircraft at a moment in time.
    """

    def __init__(self,
        rec_time: float,
        flt_key: int,
        bcn_code: int,
        cid: str,
        source: str,
        msg_type: str,
        acid: str,
        rec_type_cat: int,
        coord_1: float,
        coord_2: float,
        alt: float,
        significance: int,
        coord_1_accuracy: float,
        coord_2_accuracy: float,
        alt_accuracy: float,
        ground_speed: float,
        course: int,
        rate_of_climb: int,
        alt_qualifier: str,
        alt_indicator: str,
        track_point_status: str,
        leader_direction: int
        ) -> None:
        self.rec_time = rec_time
        self.flt_key = flt_key
        self.bcn_code = bcn_code
        self.cid = cid
        self.source = source
        self.msg_type = msg_type
        self.acid = acid
        self.rec_type_cat = rec_type_cat
        self.coord_1 = coord_1
        self.coord_2 = coord_2
        self.alt = alt
        self.significance = significance
        self.coord_1_accuracy = coord_1_accuracy
        self.coord_2_accuracy = coord_2_accuracy
        self.alt_accuracy = alt_accuracy
        self.ground_speed = ground_speed
        self.course = course
        self.rate_of_climb = rate_of_climb
        self.alt_qualifier = alt_qualifier
        self.alt_indicator = alt_indicator
        self.track_point_status = track_point_status
        self.leader_direction = leader_direction

    @staticmethod
    def from_line(line: list[str]):
        """
        Parses a Type 3 track point line from the CSV.

        Args:
            line (list[str]): Tokenized CSV line.

        Returns:
            TrackPoint: The parsed TrackPoint instance.
        """

        separated = iter(line)

        # Record type, we know this is 3 already
        _ = next(separated)
        rec_time = float(next(separated))
        flt_key = int(next(separated))
        bcn_code_ = next(separated)
        bcn_code = None if not bcn_code_ else int(bcn_code_)
        cid = next(separated)
        source = next(separated)
        msg_type = next(separated)
        acid = next(separated)
        rec_type_cat = int(next(separated))
        coord_1 = float(next(separated))
        coord_2 = float(next(separated))
        alt = float(next(separated))
        significance = int(next(separated))
        coord_1_accuracy = float(next(separated))
        coord_2_accuracy = float(next(separated))
        alt_accuracy_ = next(separated)
        alt_accuracy = None if not alt_accuracy_ else float(alt_accuracy_)
        ground_speed_ = next(separated)
        ground_speed = None if not ground_speed_ else int(ground_speed_)
        course = int(next(separated))
        rate_of_climb = int(next(separated))
        alt_qualifier = next(separated)
        alt_indicator = next(separated)
        track_point_status = next(separated)
        leader_direction_ = next(separated)
        leader_direction = None if not leader_direction_ else int(leader_direction_)

        track_point = TrackPoint(
            rec_time,
            flt_key,
            bcn_code,
            cid,
            source,
            msg_type,
            acid,
            rec_type_cat,
            coord_1,
            coord_2,
            alt,
            significance,
            coord_1_accuracy,
            coord_2_accuracy,
            alt_accuracy,
            ground_speed,
            course,
            rate_of_climb,
            alt_qualifier,
            alt_indicator,
            track_point_status,
            leader_direction
        )

        return track_point


class Flight:
    """
    Represents a single flight composed of header, flight plan, and track points.
    """

    def __init__(self, header: Header, flight_plan: list[FlightPlan], track_points: list[TrackPoint]):
        self.header = header
        self.flight_plan = flight_plan
        self.track_points = track_points


class FlightsReader:
    """
    Reads and yields complete flights from a CSV file, one at a time.
    """

    def __init__(self, path: str):
        self.path = path
        self.reader = CSVReader(path, has_header=False)
    
    def __enter__(self):
        self.reader.open()
        return self
    
    def __exit__(self, exc_type, exc_value, tb):
        self.reader.close()

    def __iter__(self):
        return self
    
    def __next__(self):
        header = None
        flight_plan = []
        track_points = []

        while True:
            try:
                # Check if this is the start of a new flight or not
                if header is not None:
                    next_line = peek(self.reader)

                    if next_line[0] == '2':
                        # We have a new header, finish this object
                        return Flight(header, flight_plan, track_points)

                line = next(self.reader)

                if line[0] == '2':
                    header = Header.from_line(line)
                elif line[0] == '4':
                    flight_plan_entry = FlightPlan.from_line(line)
                    flight_plan.append(flight_plan_entry)
                elif line[0] == '3':
                    track_point = TrackPoint.from_line(line)
                    track_points.append(track_point)
                else:
                    # Ignore the line and move on
                    pass
            except StopIteration:
                if header is None:
                    raise StopIteration
                else:
                    return Flight(header, flight_plan, track_points)


def read_flights(path: str) -> FlightsReader:
    """
    Returns a context-managed `FlightsReader` for reading flight data.

    Args:
        path (str): Path to the CSV file.

    Returns:
        FlightsReader: A reader object for iterating through flights.
    """
    return FlightsReader(path)