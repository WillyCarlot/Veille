import random
import numpy as np



class Game:

    #on établi nos actions possibles et on les met dans une table ACTIONS
    HAUT = 0
    GAUCHE = 1
    BAS = 2
    DROITE = 3

    ACTIONS = [BAS, GAUCHE, DROITE, HAUT]
    #on défini les déplacement d'un un espace 2d en lien avec nos actions 
    MOUVEMENTS = {
        HAUT: (1, 0),
        DROITE: (0, 1),
        GAUCHE: (0, -1),
        BAS: (-1, 0)
    }

    num_actions = len(ACTIONS)
    
    def __init__(self, n, m, aleatoire=False):
        self.n = n
        self.m = m
        self.aleatoire = aleatoire
        self.generer_jeu()

    def _robot_to_id(self, x, y):
        """Donne l'identifiant de la robot entre 0 et 15"""
        return x + y * self.n

    def _id_to_robot(self, id):
        """Réciproque de la fonction précédente"""
        return (id % self.n, id // self.n)

    def generer_jeu(self):
        cases = [(x, y) for x in range(self.n) for y in range(self.m)]
        arbre = random.choice(cases)
        cases.remove(arbre)
        depart = random.choice(cases)
        cases.remove(depart)
        maison = random.choice(cases)
        cases.remove(maison)


        self.robot = depart
        self.maison = maison
        self.arbre = arbre
        self.compteur = 0
        
        if not self.aleatoire:
            self.depart = depart
        return self._get_etat()
    
    def reinitialiser(self):
        if not self.aleatoire:
            self.robot = self.depart
            self.compteur = 0
            return self._get_etat()
        else:
            return self.generer_jeu()

    def _get_grille(self, x, y):
        grille = [
            [0] * self.n for i in range(self.m)
        ]
        grille[x][y] = 1
        return grille

    def _get_etat(self):
        if self.aleatoire:
            return [self._get_grille(x, y) for (x, y) in
                    [self.robot, self.maison, self.arbre]]
        return self._robot_to_id(*self.robot)

    def deplacer(self, action):
        self.compteur += 1

        if action not in self.ACTIONS:
            raise Exception("Action invalide")


        d_x, d_y = self.MOUVEMENTS[action]
        x, y = self.robot
        new_x, new_y = x + d_x, y + d_y

       
        if self.arbre == (new_x, new_y):
            self.robot = new_x, new_y
            return self._get_etat(), -10, True, None
        elif self.maison == (new_x, new_y):
            self.robot = new_x, new_y
            return self._get_etat(), 10, True, self.ACTIONS
        elif new_x >= self.n or new_y >= self.m or new_x < 0 or new_y < 0:
            return self._get_etat(), -1, False, self.ACTIONS
        elif self.compteur > 190:
            self.robot = new_x, new_y
            return self._get_etat(), -10, True, self.ACTIONS
        else:
            self.robot = new_x, new_y
            return self._get_etat(), -1, False, self.ACTIONS

    def afficher(self):
        str_jeu = ""
        for i in range(self.n - 1, -1, -1):
            for j in range(self.m):
                if (i, j) == self.robot:
                    str_jeu += " 🤖"
                elif (i, j) == self.arbre:
                    str_jeu += " 🌲"
                elif (i, j) == self.maison:
                    str_jeu += " 🏠"
                else:
                    str_jeu += " ◻️ "
            str_jeu += "\n"
        print(str_jeu)

    def train(self, num_episodes=1000, lr=0.85, y=0.99):
        etats_n = self.n * self.m
        actions_n = self.num_actions
        Q = np.zeros([etats_n, actions_n])

        cumul_reward_list = []

        for i in range(num_episodes):
            actions = []
            s = self.reinitialiser()
            cumul_reward = 0
            d = False

            while True:
                Q2 = Q[s, :] + np.random.randn(1, actions_n) * (1. / (i + 1))
                a = np.argmax(Q2)
                s1, reward, d, _ = self.deplacer(a)
                Q[s, a] = Q[s, a] + lr * (reward + y * np.max(Q[s1, :]) - Q[s, a])
                cumul_reward += reward
                s = s1
                actions.append(a)
                if d:
                    break

            cumul_reward_list.append(cumul_reward)

        #print("Score au fil du temps : " + str(sum(cumul_reward_list[-100:]) / 100.0))

        # Sauvegarde du modèle dans un fichier
        np.save('modele_table_q.npy', Q)
        self.modele = Q  # Stocker le modèle dans l'attribut de la classe

    def run_agent(self):
        s = self.reinitialiser()
        d = False

        while not d:
            a = np.argmax(self.modele[s, :])
            s, _, d, _ = self.deplacer(a)
            self.afficher()

# Exemple d'utilisation
Game = Game(3, 3, 0)
Game.train()
Game.run_agent()
