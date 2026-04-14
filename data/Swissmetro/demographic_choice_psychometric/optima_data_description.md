# Optima Reduced Official-Style HCM Data

This experiment uses the official public `optima.dat` distributed with Biogeme.

## Human sample construction

The cleaning logic follows the current official Optima family closely:

- drop observations with `Choice == -1`
- drop inconsistent rows where `Choice == 1` but `CarAvail == 3`
- keep the worker sample used in the current official family, defined by `OccupStat in {1, 2}`
- drop single-trip respondents
- drop rows with zero PT time, zero car time, or zero distance
- keep respondents with valid responses on the six selected indicators

For PT time, the raw field `TimePT` is the total PT time. It already includes walking time and waiting time. The choice scripts therefore use a derived field equal to `TimePT - WaitingTimePT` when they need PT travel time and PT waiting time as two separate attributes.

## Selected latent variables

- `car_centric_attitude`
- `environmental_attitude`

## Selected indicators

- `Envir01`
- `Mobil05`
- `LifSty07`
- `Envir05`
- `Mobil12`
- `LifSty01`

The indicator texts are taken from the public Optima description whenever recoverable.

## Choice task

The choice model uses three alternatives:

- `PT`
- `CAR`
- `SLOW_MODES`

The reduced official-style utility specification keeps:

- PT travel time excluding waiting time
- PT waiting time with work / other splits
- common monetary cost
- car time
- slow-mode distance with work / other splits
- additive effects of the two latent variables on PT and CAR
