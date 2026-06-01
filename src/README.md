# Database Queries

## approach table
WHERE Airport_Identifier LIKE 'KJFK' AND Procedure_Identifier LIKE 'R04LY'
--> https://resources.globalair.com/dtpp/globalair_00610RY4L.PDF

## terminal_waypoint table
WHERE Airport_Identifier LIKE 'KJFK' AND Waypoint_Identifier like 'REPRE'

## runway
WHERE Airport_Identifier LIKE 'KJFK'

## approch_cont
WHERE Airport_Identifier LIKE 'KJFK' AND Procedure_Identifier LIKE 'R04LY'

## localizer (only for ILS)
WHERE Runway_Identifier LIKE 'RW04L' AND Airport_Identifier LIKE 'KJFK'

## pathpoint_cont and pahtpoint
Airport_Identifier LIKE 'KJFK' AND Approach_Identifier LIKE 'R04LY'
--> Approach_Type_Identifier