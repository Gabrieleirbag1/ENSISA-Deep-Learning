Dossier contenant l'ensemble des étapes pour le TD de mise en production d'un réseau

L'idée générale est de prendre un réseau non personnel et de le transformer

Le réseau provient d'un collab de google
cp1 : On va l'importer et le containeriser (pas de python sur la machine)
cp2 : On va le structurer en deux parties : configuration, classification
cp3 : On va mettre un serveur web pour l'utiliser gràce à des routes et en concervant l'exécutable actif
cp4 : On va mettre un cache pour utiliser le réseaux en local 
cp5 : On va construire un frontal en springboot qui va rediriger les requêtes vers le container
cp6 : On va containeriser le frontal et déployer avec docker compose
cp7 : On va construire le java dans un container (java et maven disparaissent de la machine hôte)

Il faudra des outils sur la machine et des images à précharger pour gagner du temps
lors des démos.

preload est un script qui va tenter de charger les besoins
deux options exclusives et une obligatoire:
-d : on utilise le docker présent et au pire on charge un docker obsolete (cp7 ne sera pas testable)
-e : on force l'utilisation de docker engine, si une version obsolete est présente on la retire

par defaut on fait aucune option et on affiche le help avec un suicide

