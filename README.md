# madepro

A minimal Rust library crate for solving finite deterministic Markov decision processes (MDPs).

## Limitations

This library is still in development and is not ready for production use.
It only implements a few algorithms and one environment.
It is also limited to deterministic MDPs.
Originally, I only created this library in order to refactor my blog post
on [Markov Decision Processes](https://devspaceship.com/posts/gridworld).
However, I decided to probably continue working on it when I have the time.
In the future I would first like to make the library more generic
and then add more algorithms and environments.

## Features

The library currently supports the following algorithms:

- [Policy Iteration](https://en.wikipedia.org/wiki/Markov_decision_process#Policy_iteration)
- [Value Iteration](https://en.wikipedia.org/wiki/Markov_decision_process#Value_iteration)
- [SARSA](https://en.wikipedia.org/wiki/State-Action-Reward-State-Action)
- [Q-Learning](https://en.wikipedia.org/wiki/Q-learning)

## Roadmap

The following features are planned for the future:

- Stochastic MDPs
- Stochastic policies
- State-dependent action spaces
- More algorithms
