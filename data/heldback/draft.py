"""Source of the held back set. Questions were written for this project (not taken from BIRD) on three
BIRD dev databases, to check whether BIRD dev scores are inflated by pretraining contamination.
`python data/heldback/draft.py` validates every gold query and writes heldback.json."""
import json
import sqlite3
import sys
from pathlib import Path

S, M, C = "simple", "moderate", "challenging"
Q = {
"superhero": [
 ("How many superheroes have blue eyes?", "", "SELECT COUNT(*) FROM superhero s JOIN colour c ON s.eye_colour_id = c.id WHERE c.colour = 'Blue'", S),
 ("List the superhero names of all heroes published by Dark Horse Comics.", "", "SELECT s.superhero_name FROM superhero s JOIN publisher p ON s.publisher_id = p.id WHERE p.publisher_name = 'Dark Horse Comics'", S),
 ("Which publisher has the largest number of superheroes?", "Ignore heroes whose publisher name is empty.", "SELECT p.publisher_name FROM superhero s JOIN publisher p ON s.publisher_id = p.id WHERE p.publisher_name <> '' GROUP BY p.id ORDER BY COUNT(*) DESC LIMIT 1", M),
 ("What is the average height of female superheroes with a recorded height?", "female refers to gender = 'Female'; recorded height refers to height_cm > 0", "SELECT AVG(s.height_cm) FROM superhero s JOIN gender g ON s.gender_id = g.id WHERE g.gender = 'Female' AND s.height_cm > 0", M),
 ("How many superheroes can fly?", "can fly refers to power_name = 'Flight'", "SELECT COUNT(DISTINCT hp.hero_id) FROM hero_power hp JOIN superpower sp ON hp.power_id = sp.id WHERE sp.power_name = 'Flight'", M),
 ("What is the name of the heaviest superhero?", "heaviest refers to MAX(weight_kg)", "SELECT superhero_name FROM superhero WHERE weight_kg = (SELECT MAX(weight_kg) FROM superhero)", S),
 ("How many superheroes have an Intelligence value of 100?", "", "SELECT COUNT(*) FROM hero_attribute ha JOIN attribute a ON ha.attribute_id = a.id WHERE a.attribute_name = 'Intelligence' AND ha.attribute_value = 100", M),
 ("How many villains are published by Marvel Comics?", "villains refers to alignment = 'Bad'", "SELECT COUNT(*) FROM superhero s JOIN alignment a ON s.alignment_id = a.id JOIN publisher p ON s.publisher_id = p.id WHERE a.alignment = 'Bad' AND p.publisher_name = 'Marvel Comics'", M),
 ("What percentage of DC Comics superheroes are female?", "percentage = COUNT(gender = 'Female') * 100 / COUNT(*) for publisher_name = 'DC Comics'", "SELECT CAST(SUM(CASE WHEN g.gender = 'Female' THEN 1 ELSE 0 END) AS REAL) * 100 / COUNT(*) FROM superhero s JOIN publisher p ON s.publisher_id = p.id JOIN gender g ON s.gender_id = g.id WHERE p.publisher_name = 'DC Comics'", C),
 ("List all the powers of Wolverine.", "", "SELECT sp.power_name FROM superhero s JOIN hero_power hp ON s.id = hp.hero_id JOIN superpower sp ON hp.power_id = sp.id WHERE s.superhero_name = 'Wolverine'", M),
 ("Which race has the most superheroes?", "Do not count the placeholder race '-'.", "SELECT r.race FROM superhero s JOIN race r ON s.race_id = r.id WHERE r.race <> '-' GROUP BY r.id ORDER BY COUNT(*) DESC LIMIT 1", M),
 ("How many superheroes have the same eye and hair colour?", "Exclude heroes whose colour is 'No Colour'.", "SELECT COUNT(*) FROM superhero s JOIN colour c ON s.eye_colour_id = c.id WHERE s.eye_colour_id = s.hair_colour_id AND c.colour <> 'No Colour'", M),
 ("How many superheroes have more than 30 powers?", "", "SELECT COUNT(*) FROM (SELECT hero_id FROM hero_power GROUP BY hero_id HAVING COUNT(*) > 30)", M),
 ("Which superhero has the most powers?", "", "SELECT s.superhero_name FROM superhero s JOIN hero_power hp ON s.id = hp.hero_id GROUP BY s.id ORDER BY COUNT(*) DESC LIMIT 1", M),
 ("What is the average weight of superheroes with green skin?", "green skin refers to skin colour = 'Green'", "SELECT AVG(s.weight_kg) FROM superhero s JOIN colour c ON s.skin_colour_id = c.id WHERE c.colour = 'Green'", M),
 ("List the names of neutral superheroes published by Marvel Comics.", "neutral refers to alignment = 'Neutral'", "SELECT s.superhero_name FROM superhero s JOIN alignment a ON s.alignment_id = a.id JOIN publisher p ON s.publisher_id = p.id WHERE a.alignment = 'Neutral' AND p.publisher_name = 'Marvel Comics'", M),
 ("How many powers does Superman have?", "", "SELECT COUNT(*) FROM superhero s JOIN hero_power hp ON s.id = hp.hero_id WHERE s.superhero_name = 'Superman'", S),
 ("What colour are Batman's eyes?", "Batman refers to superhero_name = 'Batman'", "SELECT c.colour FROM superhero s JOIN colour c ON s.eye_colour_id = c.id WHERE s.superhero_name = 'Batman'", S),
 ("Which attribute has the highest average value across all heroes?", "", "SELECT a.attribute_name FROM hero_attribute ha JOIN attribute a ON ha.attribute_id = a.id GROUP BY a.id ORDER BY AVG(ha.attribute_value) DESC LIMIT 1", M),
 ("How many superheroes are taller than 200 cm?", "", "SELECT COUNT(*) FROM superhero WHERE height_cm > 200", S),
 ("Which publisher's superheroes have the highest average Strength?", "Only consider publishers with at least 10 heroes that have a Strength value.", "SELECT p.publisher_name FROM superhero s JOIN publisher p ON s.publisher_id = p.id JOIN hero_attribute ha ON s.id = ha.hero_id JOIN attribute a ON ha.attribute_id = a.id WHERE a.attribute_name = 'Strength' GROUP BY p.id HAVING COUNT(*) >= 10 ORDER BY AVG(ha.attribute_value) DESC LIMIT 1", C),
 ("Name the superheroes that have both Telepathy and Telekinesis.", "", "SELECT s.superhero_name FROM superhero s JOIN hero_power hp ON s.id = hp.hero_id JOIN superpower sp ON hp.power_id = sp.id WHERE sp.power_name = 'Telepathy' INTERSECT SELECT s.superhero_name FROM superhero s JOIN hero_power hp ON s.id = hp.hero_id JOIN superpower sp ON hp.power_id = sp.id WHERE sp.power_name = 'Telekinesis'", C),
 ("What is the ratio of good superheroes to bad superheroes?", "ratio = COUNT(alignment = 'Good') / COUNT(alignment = 'Bad')", "SELECT CAST(SUM(CASE WHEN a.alignment = 'Good' THEN 1 ELSE 0 END) AS REAL) / SUM(CASE WHEN a.alignment = 'Bad' THEN 1 ELSE 0 END) FROM superhero s JOIN alignment a ON s.alignment_id = a.id", C),
 ("How many superheroes have no recorded powers?", "", "SELECT COUNT(*) FROM superhero WHERE id NOT IN (SELECT hero_id FROM hero_power)", M),
 ("List the five tallest female superheroes, tallest first.", "female refers to gender = 'Female'", "SELECT s.superhero_name FROM superhero s JOIN gender g ON s.gender_id = g.id WHERE g.gender = 'Female' ORDER BY s.height_cm DESC LIMIT 5", M),
 ("What is the most common hair colour among male superheroes?", "Do not count 'No Colour'.", "SELECT c.colour FROM superhero s JOIN gender g ON s.gender_id = g.id JOIN colour c ON s.hair_colour_id = c.id WHERE g.gender = 'Male' AND c.colour <> 'No Colour' GROUP BY c.id ORDER BY COUNT(*) DESC LIMIT 1", M),
 ("How many superheroes have a Durability value below 20?", "", "SELECT COUNT(*) FROM hero_attribute ha JOIN attribute a ON ha.attribute_id = a.id WHERE a.attribute_name = 'Durability' AND ha.attribute_value < 20", M),
 ("How many superheroes use their full name as their superhero name?", "", "SELECT COUNT(*) FROM superhero WHERE full_name = superhero_name", S),
 ("How many more superheroes does Marvel Comics have than DC Comics?", "", "SELECT SUM(CASE WHEN p.publisher_name = 'Marvel Comics' THEN 1 ELSE 0 END) - SUM(CASE WHEN p.publisher_name = 'DC Comics' THEN 1 ELSE 0 END) FROM superhero s JOIN publisher p ON s.publisher_id = p.id", C),
 ("How many power assignments belong to mutant superheroes?", "mutant refers to race = 'Mutant'", "SELECT COUNT(*) FROM hero_power hp JOIN superhero s ON hp.hero_id = s.id JOIN race r ON s.race_id = r.id WHERE r.race = 'Mutant'", M),
 ("Which gender has the highest average superhero weight?", "Only count heroes with weight_kg > 0.", "SELECT g.gender FROM superhero s JOIN gender g ON s.gender_id = g.id WHERE s.weight_kg > 0 GROUP BY g.id ORDER BY AVG(s.weight_kg) DESC LIMIT 1", M),
 ("List the superheroes heavier than 500 kg together with their publisher.", "", "SELECT s.superhero_name, p.publisher_name FROM superhero s JOIN publisher p ON s.publisher_id = p.id WHERE s.weight_kg > 500", M),
 ("Which superheroes have the maximum possible Speed value of 100? Give their superhero names.", "", "SELECT s.superhero_name FROM superhero s JOIN hero_attribute ha ON s.id = ha.hero_id JOIN attribute a ON ha.attribute_id = a.id WHERE a.attribute_name = 'Speed' AND ha.attribute_value = 100", M),
 ("What share of superheroes published by Marvel Comics are good, as a percentage?", "good refers to alignment = 'Good'", "SELECT CAST(SUM(CASE WHEN a.alignment = 'Good' THEN 1 ELSE 0 END) AS REAL) * 100 / COUNT(*) FROM superhero s JOIN publisher p ON s.publisher_id = p.id JOIN alignment a ON s.alignment_id = a.id WHERE p.publisher_name = 'Marvel Comics'", C),
],
"student_club": [
 ("How many members wear a Large t-shirt?", "", "SELECT COUNT(*) FROM member WHERE t_shirt_size = 'Large'", S),
 ("List the names of the events held in MU 215.", "", "SELECT event_name FROM event WHERE location = 'MU 215'", S),
 ("How much money has been spent in total on the Food budget category?", "", "SELECT SUM(spent) FROM budget WHERE category = 'Food'", S),
 ("How many events were games?", "games refers to type = 'Game'", "SELECT COUNT(*) FROM event WHERE type = 'Game'", S),
 ("Which member attended the most events? Give the first and last name.", "", "SELECT m.first_name, m.last_name FROM attendance a JOIN member m ON a.link_to_member = m.member_id GROUP BY m.member_id ORDER BY COUNT(*) DESC LIMIT 1", M),
 ("How much income has the club received from dues in total?", "dues refers to source = 'Dues'", "SELECT SUM(amount) FROM income WHERE source = 'Dues'", S),
 ("What is the email address of the club president?", "president refers to position = 'President'", "SELECT email FROM member WHERE position = 'President'", S),
 ("How many members live in Illinois?", "", "SELECT COUNT(*) FROM member m JOIN zip_code z ON m.zip = z.zip_code WHERE z.state = 'Illinois'", M),
 ("Which event had the highest total expense cost?", "total expense cost = SUM(cost) of expenses linked to the event's budgets", "SELECT e.event_name FROM expense x JOIN budget b ON x.link_to_budget = b.budget_id JOIN event e ON b.link_to_event = e.event_id GROUP BY e.event_id ORDER BY SUM(x.cost) DESC LIMIT 1", C),
 ("What is the average cost of pizza expenses?", "pizza expenses refers to expense_description = 'Pizza'", "SELECT AVG(cost) FROM expense WHERE expense_description = 'Pizza'", S),
 ("List the full names of members who paid expenses from an Advertisement budget.", "full name refers to first_name, last_name", "SELECT DISTINCT m.first_name, m.last_name FROM expense x JOIN budget b ON x.link_to_budget = b.budget_id JOIN member m ON x.link_to_member = m.member_id WHERE b.category = 'Advertisement'", C),
 ("Which budget category has the lowest total remaining amount?", "", "SELECT category FROM budget GROUP BY category ORDER BY SUM(remaining) ASC LIMIT 1", M),
 ("How many events took place in 2019?", "2019 refers to SUBSTR(event_date, 1, 4) = '2019'", "SELECT COUNT(*) FROM event WHERE SUBSTR(event_date, 1, 4) = '2019'", S),
 ("Which major has the most club members?", "", "SELECT mj.major_name FROM member m JOIN major mj ON m.link_to_major = mj.major_id GROUP BY mj.major_id ORDER BY COUNT(*) DESC LIMIT 1", M),
 ("List the full names of members who have never attended an event.", "full name refers to first_name, last_name", "SELECT first_name, last_name FROM member WHERE member_id NOT IN (SELECT link_to_member FROM attendance)", M),
 ("What is the total budgeted amount for the October Meeting?", "budgeted amount refers to SUM(amount) of the event's budgets", "SELECT SUM(b.amount) FROM budget b JOIN event e ON b.link_to_event = e.event_id WHERE e.event_name = 'October Meeting'", M),
 ("How many expenses were made in September 2019?", "September 2019 refers to expense_date LIKE '2019-09%'", "SELECT COUNT(*) FROM expense WHERE expense_date LIKE '2019-09%'", S),
 ("What percentage of events are closed?", "closed refers to status = 'Closed'", "SELECT CAST(SUM(CASE WHEN status = 'Closed' THEN 1 ELSE 0 END) AS REAL) * 100 / COUNT(*) FROM event", M),
 ("Who made the single most expensive expense? Give the first and last name.", "", "SELECT m.first_name, m.last_name FROM expense x JOIN member m ON x.link_to_member = m.member_id ORDER BY x.cost DESC LIMIT 1", M),
 ("In which city does the club's Vice President live?", "Vice President refers to position = 'Vice President'", "SELECT z.city FROM member m JOIN zip_code z ON m.zip = z.zip_code WHERE m.position = 'Vice President'", M),
 ("How many budgets have been overspent?", "overspent refers to remaining < 0", "SELECT COUNT(*) FROM budget WHERE remaining < 0", S),
 ("How many members are there for each t-shirt size?", "", "SELECT t_shirt_size, COUNT(*) FROM member GROUP BY t_shirt_size", M),
 ("Which events had a budget for speaker gifts?", "speaker gifts refers to category = 'Speaker Gifts'", "SELECT DISTINCT e.event_name FROM budget b JOIN event e ON b.link_to_event = e.event_id WHERE b.category = 'Speaker Gifts'", M),
 ("How much income did the club receive in October 2019?", "October 2019 refers to date_received LIKE '2019-10%'", "SELECT SUM(amount) FROM income WHERE date_received LIKE '2019-10%'", S),
 ("Which event had the most attendees?", "", "SELECT e.event_name FROM attendance a JOIN event e ON a.link_to_event = e.event_id GROUP BY e.event_id ORDER BY COUNT(*) DESC LIMIT 1", M),
 ("On average, how many members attended each event that had at least one attendee?", "", "SELECT CAST(COUNT(*) AS REAL) / COUNT(DISTINCT link_to_event) FROM attendance", C),
 ("What are the phone numbers of the members who paid dues in October 2019?", "dues refers to source = 'Dues'; October 2019 refers to date_received LIKE '2019-10%'", "SELECT DISTINCT m.phone FROM income i JOIN member m ON i.link_to_member = m.member_id WHERE i.source = 'Dues' AND i.date_received LIKE '2019-10%'", M),
 ("Which department is Grant Gilmour's major in?", "", "SELECT mj.department FROM member m JOIN major mj ON m.link_to_major = mj.major_id WHERE m.first_name = 'Grant' AND m.last_name = 'Gilmour'", M),
 ("How many open meetings are there?", "open meetings refers to type = 'Meeting' AND status = 'Open'", "SELECT COUNT(*) FROM event WHERE type = 'Meeting' AND status = 'Open'", S),
 ("What is the total cost of expenses paid by members of the College of Humanities and Social Sciences?", "", "SELECT SUM(x.cost) FROM expense x JOIN member m ON x.link_to_member = m.member_id JOIN major mj ON m.link_to_major = mj.major_id WHERE mj.college = 'College of Humanities and Social Sciences'", C),
 ("How many members live in a state other than the one most members live in?", "", "SELECT COUNT(*) FROM member m JOIN zip_code z ON m.zip = z.zip_code WHERE z.state <> (SELECT z2.state FROM member m2 JOIN zip_code z2 ON m2.zip = z2.zip_code GROUP BY z2.state ORDER BY COUNT(*) DESC LIMIT 1)", C),
 ("List each budget category with the total amount spent on it, largest first.", "", "SELECT category, SUM(spent) FROM budget GROUP BY category ORDER BY SUM(spent) DESC", M),
 ("Which events in 2020 were elections?", "2020 refers to SUBSTR(event_date, 1, 4) = '2020'; elections refers to type = 'Election'", "SELECT event_name FROM event WHERE type = 'Election' AND SUBSTR(event_date, 1, 4) = '2020'", S),
],
"formula_1": [
 ("How many races were held in 2010?", "", "SELECT COUNT(*) FROM races WHERE year = 2010", S),
 ("Which country has the most circuits?", "", "SELECT country FROM circuits GROUP BY country ORDER BY COUNT(*) DESC LIMIT 1", M),
 ("What is Fernando Alonso's nationality?", "", "SELECT nationality FROM drivers WHERE forename = 'Fernando' AND surname = 'Alonso'", S),
 ("How many drivers were born after 1990?", "born after 1990 refers to STRFTIME('%Y', dob) > '1990'", "SELECT COUNT(*) FROM drivers WHERE STRFTIME('%Y', dob) > '1990'", S),
 ("At which circuit was the 2015 Monaco Grand Prix held?", "", "SELECT c.name FROM races r JOIN circuits c ON r.circuitId = c.circuitId WHERE r.year = 2015 AND r.name = 'Monaco Grand Prix'", M),
 ("Which driver has won the most races? Give the forename and surname.", "won refers to position = 1 in results", "SELECT d.forename, d.surname FROM results r JOIN drivers d ON r.driverId = d.driverId WHERE r.position = 1 GROUP BY d.driverId ORDER BY COUNT(*) DESC LIMIT 1", M),
 ("How many races has Lewis Hamilton won?", "won refers to position = 1", "SELECT COUNT(*) FROM results r JOIN drivers d ON r.driverId = d.driverId WHERE d.forename = 'Lewis' AND d.surname = 'Hamilton' AND r.position = 1", M),
 ("Which constructor scored the most points in the 2012 season?", "points come from constructorResults", "SELECT c.name FROM constructorResults cr JOIN races r ON cr.raceId = r.raceId JOIN constructors c ON cr.constructorId = c.constructorId WHERE r.year = 2012 GROUP BY c.constructorId ORDER BY SUM(cr.points) DESC LIMIT 1", C),
 ("Who set the fastest lap of the 2009 Australian Grand Prix?", "fastest lap refers to rank = 1 in results", "SELECT d.forename, d.surname FROM results res JOIN races r ON res.raceId = r.raceId JOIN drivers d ON res.driverId = d.driverId WHERE r.year = 2009 AND r.name = 'Australian Grand Prix' AND res.rank = 1", M),
 ("How many pit stops did Sebastian Vettel make in the 2011 Chinese Grand Prix?", "", "SELECT COUNT(*) FROM pitStops p JOIN races r ON p.raceId = r.raceId JOIN drivers d ON p.driverId = d.driverId WHERE r.year = 2011 AND r.name = 'Chinese Grand Prix' AND d.forename = 'Sebastian' AND d.surname = 'Vettel'", C),
 ("What was the average pit stop time in milliseconds at the 2012 Malaysian Grand Prix?", "", "SELECT AVG(p.milliseconds) FROM pitStops p JOIN races r ON p.raceId = r.raceId WHERE r.year = 2012 AND r.name = 'Malaysian Grand Prix'", M),
 ("List the names of the circuits in Italy.", "", "SELECT name FROM circuits WHERE country = 'Italy'", S),
 ("What is Max Verstappen's driver code?", "", "SELECT code FROM drivers WHERE forename = 'Max' AND surname = 'Verstappen'", S),
 ("How many races have been held at Silverstone Circuit?", "", "SELECT COUNT(*) FROM races r JOIN circuits c ON r.circuitId = c.circuitId WHERE c.name = 'Silverstone Circuit'", M),
 ("Which driver started from pole position the most times in qualifying?", "pole position refers to qualifying position = 1", "SELECT d.forename, d.surname FROM qualifying q JOIN drivers d ON q.driverId = d.driverId WHERE q.position = 1 GROUP BY d.driverId ORDER BY COUNT(*) DESC LIMIT 1", M),
 ("How many constructors are German?", "German refers to nationality = 'German'", "SELECT COUNT(*) FROM constructors WHERE nationality = 'German'", S),
 ("Apart from finishing, what is the most common race result status?", "Exclude 'Finished' and statuses that start with '+' (lapped cars).", "SELECT s.status FROM results r JOIN status s ON r.statusId = s.statusId WHERE s.status <> 'Finished' AND s.status NOT LIKE '+%' GROUP BY s.statusId ORDER BY COUNT(*) DESC LIMIT 1", C),
 ("Which season had the most races?", "", "SELECT year FROM races GROUP BY year ORDER BY COUNT(*) DESC LIMIT 1", M),
 ("Which drivers raced for Ferrari in 2008? Give forename and surname.", "", "SELECT DISTINCT d.forename, d.surname FROM results res JOIN races r ON res.raceId = r.raceId JOIN constructors c ON res.constructorId = c.constructorId JOIN drivers d ON res.driverId = d.driverId WHERE c.name = 'Ferrari' AND r.year = 2008", C),
 ("How many points did Michael Schumacher score over his career?", "career points = SUM(points) in results", "SELECT SUM(r.points) FROM results r JOIN drivers d ON r.driverId = d.driverId WHERE d.forename = 'Michael' AND d.surname = 'Schumacher'", M),
 ("When was the first race held at Monza?", "Monza refers to circuitRef = 'monza'", "SELECT MIN(r.date) FROM races r JOIN circuits c ON r.circuitId = c.circuitId WHERE c.circuitRef = 'monza'", M),
 ("How many laps did the winner of the 2014 Italian Grand Prix complete?", "winner refers to position = 1", "SELECT res.laps FROM results res JOIN races r ON res.raceId = r.raceId WHERE r.year = 2014 AND r.name = 'Italian Grand Prix' AND res.position = 1", M),
 ("Which circuit lies furthest north?", "furthest north refers to MAX(lat)", "SELECT name FROM circuits ORDER BY lat DESC LIMIT 1", S),
 ("What was Lewis Hamilton's Q1 time at the 2008 Australian Grand Prix?", "", "SELECT q.q1 FROM qualifying q JOIN races r ON q.raceId = r.raceId JOIN drivers d ON q.driverId = d.driverId WHERE r.year = 2008 AND r.name = 'Australian Grand Prix' AND d.forename = 'Lewis' AND d.surname = 'Hamilton'", M),
 ("How many different drivers scored points in the 2016 season?", "scored points refers to points > 0 in results", "SELECT COUNT(DISTINCT res.driverId) FROM results res JOIN races r ON res.raceId = r.raceId WHERE r.year = 2016 AND res.points > 0", C),
 ("Who recorded the quickest single lap at the 2010 Bahrain Grand Prix?", "quickest lap refers to MIN(milliseconds) in lapTimes", "SELECT d.forename, d.surname FROM lapTimes l JOIN races r ON l.raceId = r.raceId JOIN drivers d ON l.driverId = d.driverId WHERE r.year = 2010 AND r.name = 'Bahrain Grand Prix' ORDER BY l.milliseconds ASC LIMIT 1", C),
 ("What percentage of race results in 2010 had the status Finished?", "", "SELECT CAST(SUM(CASE WHEN s.status = 'Finished' THEN 1 ELSE 0 END) AS REAL) * 100 / COUNT(*) FROM results res JOIN races r ON res.raceId = r.raceId JOIN status s ON res.statusId = s.statusId WHERE r.year = 2010", C),
 ("What is the nationality of the Red Bull constructor?", "", "SELECT nationality FROM constructors WHERE name = 'Red Bull'", S),
 ("How many races in the 2005 season were held in the USA?", "", "SELECT COUNT(*) FROM races r JOIN circuits c ON r.circuitId = c.circuitId WHERE r.year = 2005 AND c.country = 'USA'", M),
 ("Who is the youngest driver in the database?", "youngest refers to the latest dob", "SELECT forename, surname FROM drivers ORDER BY dob DESC LIMIT 1", S),
 ("What was Jenson Button's championship position after the final race of 2009?", "final race refers to MAX(round) of 2009", "SELECT ds.position FROM driverStandings ds JOIN races r ON ds.raceId = r.raceId JOIN drivers d ON ds.driverId = d.driverId WHERE d.forename = 'Jenson' AND d.surname = 'Button' AND r.year = 2009 AND r.round = (SELECT MAX(round) FROM races WHERE year = 2009)", C),
 ("What was the average fastest lap speed at the 2010 Monaco Grand Prix?", "", "SELECT AVG(CAST(res.fastestLapSpeed AS REAL)) FROM results res JOIN races r ON res.raceId = r.raceId WHERE r.year = 2010 AND r.name = 'Monaco Grand Prix'", M),
 ("Which drivers have used the permanent number 44?", "", "SELECT forename, surname FROM drivers WHERE number = 44", S),
],
}


def main(db_root):
    out, bad = [], 0
    for db, qs in Q.items():
        conn = sqlite3.connect(f"file:{Path(db_root) / db / (db + '.sqlite')}?mode=ro", uri=True)
        for i, (q, ev, sql, diff) in enumerate(qs):
            try:
                rows = conn.execute(sql).fetchall()
            except sqlite3.Error as e:
                rows, bad = None, bad + 1
                print("ERROR", db, i, e)
            if rows is not None and (not rows or rows == [(None,)]):
                bad += 1
                print("EMPTY", db, i, q)
            if "-v" in sys.argv:
                print(db, i, q, "=>", rows if rows is None else rows[:3], len(rows or []))
            out.append({"id": f"{db}_{i:02d}", "db_id": db, "question": q, "evidence": ev, "sql": sql, "difficulty": diff})
    Path(__file__).with_name("heldback.json").write_text(json.dumps(out, indent=1))
    print(len(out), "questions;", bad, "problems")
    return bad


if __name__ == "__main__":
    sys.exit(main(sys.argv[1] if len(sys.argv) > 1 and sys.argv[1] != "-v" else "data/raw/dev/dev_20240627/dev_databases") > 0)
