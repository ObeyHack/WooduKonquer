# WooduKonquer

Welcome to the WooduKonquer project, this project aims to explore different AI driven strategies to 
play the game of Woodoku.

https://github.com/user-attachments/assets/444c8485-bcf0-4392-9b06-90f6459615cf

---
## How to run the project

### Environment Setup
To run the project, you need to have `python:3.10` or higher installed on your machine.
1. `clone` the project
```bash
git clone ...
```
2. Create a virtual environment
```bash
python -m venv venv
```
3. Activate the virtual environment 
   1. For **linux**:
   ```bash
    source venv/bin/activate
    ```
   2. For **Windows**:
   ```bash
    .\venv\scripts\activate
    ```

4. Install the dependencies (inside the WooduKonquer directory, the same directory as pyproject.toml)
```bash
pip install .
```
5. Run the project
```bash
woodoku-cli -h
```

## The Different Agents and Their Flags

```Usage
usage: woodoku-cli [-h] [-a {random,reflex,minimax,alpha_beta,expectimax,q_learning,q_approx}] [-d {GUI,Text,SummaryDisplay,Testing}] [-l]

Play Woodoku Game

options:
  -h, --help            show this help message and exit
  -a {random,reflex,minimax,alpha_beta,expectimax,q_learning,q_approx}, --agent {random,reflex,minimax,alpha_beta,expectimax,q_learning,q_approx}
                        Agent to play the game
  -d {GUI,Text,SummaryDisplay,Testing}, --display {GUI,Text,SummaryDisplay,Testing}
                        Render mode
  -l, --log             Log the game via neptune

```
As can be seen from the above, the project has different agents that can be used to play the game. The agents are:
1. `random` - This agent plays the game randomly. using: `woodoku-cli -a random`
2. `reflex` - This agent plays the game using a reflex agent `woodoku-cli -a reflex`
3. `minimax` - This agent plays the game using the minimax algorithm `woodoku-cli -a minimax`
4. `alpha_beta` - This agent plays the game using the alpha-beta pruning algorithm `woodoku-cli -a alpha_beta`
5. `expectimax` - This agent plays the game using the expectimax algorithm `woodoku-cli -a expectimax`
6. `q_learning` - This agent plays the game using the q-learning algorithm `woodoku-cli -a q_learning`
7. `q_approx` - This agent plays the game using the q-learning algorithm with function approximation `woodoku-cli -a q_approx`

In addition to the agents, the project has different display modes that can be used to render the game. The display modes are:
1. `GUI` - This mode renders the game using a GUI of pygame. `woodoku-cli -d GUI`
2. `Text` - This mode renders the game using terminal text `woodoku-cli -d Text`
3. `SummaryDisplay` - This doesn't render the game but only display the summary of the 
   results for `10` games. `woodoku-cli -d SummaryDisplay`
4. `Testing` - This mode doesn't render the game but only display the results of the game. `woodoku-cli -d Testing`

* The `-l` flag is used to log the game score using `neptune.ai`, which should be set up via `.env` file.

So to play the game using the `minimax` agent and render the game using the `GUI` mode, you can use the following command:
```bash
woodoku-cli -a minimax -d GUI
```
---
### Report
For more information about the project, you can check the
[report](https://drive.google.com/file/d/1iBA_bMo2xjarJwCBzh8i9g2XFQLSVJax/view?usp=sharing) for the project.

Hope you enjoy playing the game! 🎉🎉🎉
