# Portefólio de Aprendizagem por Reforço

Este repositório reúne as várias práticas desenvolvidas na unidade curricular, bem como a reorganização de exercícios e notebooks antigos numa estrutura modular única. Assim, em vez de existir código disperso por ficheiros isolados, o projeto passou a separar ambientes, MDPs, agentes, experiências, visualizações e scripts de execução.

Além disso, esta organização facilita tanto a navegação como a validação do trabalho realizado. Por um lado, torna mais simples perceber onde está cada implementação; por outro, permite executar cada prática através de scripts ou notebooks específicos, mantendo uma base de código mais consistente.

## Estrutura do projeto

- [core/](core/) contém as abstrações base do projeto, como `Environment`, `Agent`, `Policy`, `Episode` e `Transition`.
- [envs/](envs/) reúne os ambientes interativos usados sobretudo em métodos model-free, como Blackjack, Windy Gridworld, Tic-Tac-Toe, K-armed Bandits e o GridWorld base.
- [mdps/](mdps/) contém os modelos conhecidos usados em dynamic programming. Neste momento, inclui a base abstrata e o problema de Jack’s Car Rental.
- [policies/](policies/) agrega políticas reutilizáveis, tanto para ambientes específicos como para versões estocásticas ou determinísticas de alguns problemas.
- [agents/](agents/) concentra os algoritmos de aprendizagem, divididos por previsão, controlo e planeamento.
- [experiments/](experiments/) junta funções auxiliares de treino, avaliação, rollouts e rotinas de dynamic programming.
- [plots/](plots/) inclui as funções de visualização reutilizáveis para gráficos, heatmaps e políticas.
- [features/](features/) guarda representações de estado usadas em aproximação de funções, como Windy Gridworld e Tic-Tac-Toe.
- [scripts/](scripts/) contém os pontos de entrada para correr as experiências principais sem depender diretamente dos notebooks.
- [notebooks/](notebooks/) reúne os notebooks usados nas práticas e nos exercícios reorganizados.
- [PDFs/](PDFs/) guarda os enunciados e materiais em PDF usados como base para as práticas.
- [outputs/](outputs/) guarda os resultados gerados, incluindo figuras e comparações produzidas pelos scripts.

## Práticas concluídas

### Prática 4

Foi implementada a parte de previsão em Blackjack, incluindo Monte Carlo, TD(0) e o exercício opcional de `n-step TD`.

- Obrigatório: [envs/blackjack.py](envs/blackjack.py), `BlackjackEnv.step`
- Obrigatório: [agents/prediction/monte_carlo.py](agents/prediction/monte_carlo.py), `MonteCarloPrediction.update_episode`
- Obrigatório: [agents/prediction/td.py](agents/prediction/td.py), `TDPrediction.update_episode`
- Opcional: [agents/prediction/n_step_td.py](agents/prediction/n_step_td.py), `NStepTDPrediction.update_episode`
- Validação principal: [scripts/run_blackjack_prediction.py](scripts/run_blackjack_prediction.py)

### Prática 5

Nesta prática foi desenvolvido o controlo em Windy Gridworld com SARSA, bem como a extensão opcional para `n-step SARSA`.

- Obrigatório: [envs/windy_gridworld.py](envs/windy_gridworld.py), `WindyGridworldEnv.step_from_state`
- Obrigatório: [agents/control/sarsa.py](agents/control/sarsa.py), `SarsaControl.select_action` e `SarsaControl.update_transition`
- Opcional: [agents/control/n_step_sarsa.py](agents/control/n_step_sarsa.py), `NStepSarsaControl`
- Validação principal: [scripts/run_windy_gridworld_sarsa.py](scripts/run_windy_gridworld_sarsa.py) e [scripts/run_windy_gridworld_n_step_sarsa.py](scripts/run_windy_gridworld_n_step_sarsa.py)

### Prática 6

A prática centrou-se em aproximação de funções no Windy Gridworld e, além disso, incluiu a implementação do ambiente de Tic-Tac-Toe para o portefólio.

- Obrigatório: [agents/control/linear_sarsa.py](agents/control/linear_sarsa.py), `LinearSarsaControl.update_transition`
- Obrigatório: [agents/control/torch_sarsa.py](agents/control/torch_sarsa.py), `TorchSarsaControl.update_transition`
- Portefólio: [envs/tictactoe.py](envs/tictactoe.py), `TicTacToeEnv.reset`, `available_actions`, `is_terminal`, `step` e `render`
- Validação principal: [scripts/run_windy_gridworld_linear_sarsa.py](scripts/run_windy_gridworld_linear_sarsa.py), [scripts/run_windy_gridworld_torch_sarsa.py](scripts/run_windy_gridworld_torch_sarsa.py) e [notebooks/TicTacToe_Demo.ipynb](notebooks/TicTacToe_Demo.ipynb)

### Prática 7

Aqui foi explorado o Tic-Tac-Toe com codificação de estados e execução de jogos entre políticas.

- Obrigatório: [features/tictactoe.py](features/tictactoe.py), `encode_state`
- Obrigatório: [experiments/tictactoe.py](experiments/tictactoe.py), `play_game`
- Base usada nesta prática: [envs/tictactoe.py](envs/tictactoe.py)
- Validação principal: [notebooks/TicTacToe_Demo.ipynb](notebooks/TicTacToe_Demo.ipynb)

### Prática 8

Esta prática introduziu policy gradient com REINFORCE no Tic-Tac-Toe.

- Obrigatório: [agents/control/reinforce.py](agents/control/reinforce.py), `_probs`
- Obrigatório: [agents/control/reinforce.py](agents/control/reinforce.py), `update_episode`
- Obrigatório: [experiments/reinforce_tictactoe.py](experiments/reinforce_tictactoe.py), `run_reinforce_episode`
- Validação principal: [notebooks/TicTacToe_PolicyGradient.ipynb](notebooks/TicTacToe_PolicyGradient.ipynb)

### Prática 9

Por fim, foi implementado o módulo de planeamento com Monte Carlo Tree Search para Tic-Tac-Toe.

- Obrigatório: [agents/planning/mcts.py](agents/planning/mcts.py), `MCTSNode.backpropagate`
- Obrigatório: [agents/planning/mcts.py](agents/planning/mcts.py), `MCTSAgent._rollout`
- Apoio à prática: [experiments/mcts_tictactoe.py](experiments/mcts_tictactoe.py)
- Validação principal: [notebooks/TicTacToe_MCTS.ipynb](notebooks/TicTacToe_MCTS.ipynb)

## Exercícios reorganizados para a estrutura atual

Além das práticas da unidade curricular, foram também integrados alguns exercícios antigos, mantendo a ideia original mas adaptando-os à arquitetura atual do projeto.

### K-armed Bandits

O exercício standalone foi dividido por ambiente, agentes, experiências, plots, script e notebook, sem alterar a lógica base de `epsilon-greedy`, `UCB` e `gradient bandit`.

- Implementação: [envs/k_bandits.py](envs/k_bandits.py), [agents/bandits.py](agents/bandits.py), [experiments/bandits.py](experiments/bandits.py), [plots/bandits.py](plots/bandits.py)
- Execução: [scripts/run_k_bandits.py](scripts/run_k_bandits.py)
- Notebook: [notebooks/KBandits.ipynb](notebooks/KBandits.ipynb)

### GridWorld DP

O notebook original de MDP/GridWorld foi reorganizado para separar o ambiente, as políticas, os algoritmos de dynamic programming, os plots e o script de execução.

- Implementação: [envs/gridworld.py](envs/gridworld.py), [policies/gridworld.py](policies/gridworld.py), [experiments/gridworld_dp.py](experiments/gridworld_dp.py), [plots/gridworld.py](plots/gridworld.py)
- Execução: [scripts/run_gridworld_dp.py](scripts/run_gridworld_dp.py)
- Notebook: [notebooks/GridWorld_DP.ipynb](notebooks/GridWorld_DP.ipynb)

### Jack’s Car Rental

Neste caso, a reorganização foi feita em torno de um MDP conhecido, pelo que a implementação ficou em [mdps/](mdps/) e não em [envs/](envs/). Deste modo, o projeto reflete melhor a natureza do problema e dos algoritmos de dynamic programming usados.

- Implementação: [mdps/car_rental.py](mdps/car_rental.py), [experiments/car_rental_dp.py](experiments/car_rental_dp.py), [plots/car_rental.py](plots/car_rental.py)
- Execução: [scripts/run_car_rental_dp.py](scripts/run_car_rental_dp.py)
- Notebook: [notebooks/CarRental_DP.ipynb](notebooks/CarRental_DP.ipynb)

### GridWorld com Policy Improvement e Policy Iteration

Por fim, a extensão do GridWorld para policy improvement e policy iteration foi integrada sobre a base já criada para o problema anterior, reutilizando o mesmo ambiente e acrescentando as novas rotinas experimentais.

- Implementação: [experiments/gridworld_dp.py](experiments/gridworld_dp.py), [plots/gridworld.py](plots/gridworld.py)
- Execução: [scripts/run_gridworld_policy_iteration.py](scripts/run_gridworld_policy_iteration.py)
- Notebook: [notebooks/GridWorld_PolicyIteration.ipynb](notebooks/GridWorld_PolicyIteration.ipynb)
