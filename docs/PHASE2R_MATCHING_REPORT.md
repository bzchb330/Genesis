# Phase 2R matching report

Matching used nearest neighbours without replacement on standardized baseline stability covariates. No B outcomes or hypothesized resource variables were used. Calibration states were removed before matching.

- Endpoint populations: FINGERTIP 221; PALMAR_SECURED 150
- Calibration reserve: 20 per group
- Formal matched pairs: 100
- Discarded formal-pool states: `{"FINGERTIP": 101, "PALMAR_SECURED": 30}`
- Matching-distance distribution: `{"count": 100, "maximum": 3.348357771707686, "mean": 2.1983286469002543, "median": 2.324002750090145, "minimum": 0.40131343026835303, "standard_deviation": 0.7708910348449985}`

## Balance before matching

```json
{
  "ferrari_canny_epsilon": {
    "FINGERTIP": {
      "mean": 0.11893690008753247,
      "standard_deviation": 0.04060265500922895
    },
    "PALMAR_SECURED": {
      "mean": 0.12700468954959004,
      "standard_deviation": 0.05122071917646617
    },
    "standardized_mean_difference_palmar_minus_fingertip": 0.17456088480412152
  },
  "total_A_normal_force_N": {
    "FINGERTIP": {
      "mean": 4.124700125316925,
      "standard_deviation": 1.3975778256016873
    },
    "PALMAR_SECURED": {
      "mean": 4.336962579611138,
      "standard_deviation": 1.5548379955262956
    },
    "standardized_mean_difference_palmar_minus_fingertip": 0.14358545411017523
  },
  "A_translation_drift_m": {
    "FINGERTIP": {
      "mean": 0.0012694241884353217,
      "standard_deviation": 0.0004478612393372372
    },
    "PALMAR_SECURED": {
      "mean": 0.003143178160240924,
      "standard_deviation": 0.0012329137006984248
    },
    "standardized_mean_difference_palmar_minus_fingertip": 2.020135740998033
  },
  "A_rotation_drift_rad": {
    "FINGERTIP": {
      "mean": 0.034338391813855444,
      "standard_deviation": 0.018967249346473074
    },
    "PALMAR_SECURED": {
      "mean": 0.0642311431185038,
      "standard_deviation": 0.0365398009616959
    },
    "standardized_mean_difference_palmar_minus_fingertip": 1.0268502290386317
  },
  "minimum_joint_margin_rad": {
    "FINGERTIP": {
      "mean": 0.20688205650108693,
      "standard_deviation": 0.06353117217222674
    },
    "PALMAR_SECURED": {
      "mean": 0.08176562158558925,
      "standard_deviation": 0.07626098531843065
    },
    "standardized_mean_difference_palmar_minus_fingertip": -1.7826591375310301
  }
}
```

## Balance after matching

```json
{
  "ferrari_canny_epsilon": {
    "FINGERTIP": {
      "mean": 0.1143242808863501,
      "standard_deviation": 0.0420498940823379
    },
    "PALMAR_SECURED": {
      "mean": 0.12434636064984791,
      "standard_deviation": 0.045338852005174554
    },
    "standardized_mean_difference_palmar_minus_fingertip": 0.22920547060407784
  },
  "total_A_normal_force_N": {
    "FINGERTIP": {
      "mean": 4.388303026321433,
      "standard_deviation": 1.4568108428890458
    },
    "PALMAR_SECURED": {
      "mean": 4.31586093421058,
      "standard_deviation": 1.5746044557715062
    },
    "standardized_mean_difference_palmar_minus_fingertip": -0.047758196169481223
  },
  "A_translation_drift_m": {
    "FINGERTIP": {
      "mean": 0.0014871713244170891,
      "standard_deviation": 0.00042417994733609434
    },
    "PALMAR_SECURED": {
      "mean": 0.002886998124519403,
      "standard_deviation": 0.0011996032806938916
    },
    "standardized_mean_difference_palmar_minus_fingertip": 1.5558545993956658
  },
  "A_rotation_drift_rad": {
    "FINGERTIP": {
      "mean": 0.04389730476708177,
      "standard_deviation": 0.020643031687395605
    },
    "PALMAR_SECURED": {
      "mean": 0.05607275764241559,
      "standard_deviation": 0.02806018792631413
    },
    "standardized_mean_difference_palmar_minus_fingertip": 0.49428637391996233
  },
  "minimum_joint_margin_rad": {
    "FINGERTIP": {
      "mean": 0.18906064975692224,
      "standard_deviation": 0.07121436281355117
    },
    "PALMAR_SECURED": {
      "mean": 0.09186997130074664,
      "standard_deviation": 0.07606706601735515
    },
    "standardized_mean_difference_palmar_minus_fingertip": -1.3190796265551705
  }
}
```
