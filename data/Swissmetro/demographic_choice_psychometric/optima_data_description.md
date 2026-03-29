# Optima Reduced Official-Style HCM Data

This experiment uses the official public `optima.dat` distributed with Biogeme.

## Human sample construction

The cleaning logic follows the current official Optima family closely:

- drop observations with `Choice == -1`
- drop inconsistent rows where `Choice == 1` but `CarAvail == 3`
- keep the worker sample used in the current official family
- drop single-trip respondents
- drop rows with zero PT time, zero car time, or zero distance
- keep respondents with valid responses on the six selected indicators

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

- PT time
- PT waiting time with work / other splits
- common monetary cost
- car time
- slow-mode distance with work / other splits
- additive effects of the two latent variables on PT and CAR
