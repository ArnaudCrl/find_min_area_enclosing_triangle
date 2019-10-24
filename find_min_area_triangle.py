import numpy as np
from math import atan2, sqrt
import matplotlib.pyplot as plt
from random import randint


class Point:
    """ Definit les points de l'espace 2D.
        chaque point a pour attribus une abcisse x et une ordonne y

        """

    def __init__(self, x=0, y=0):
        """

        @param x: abscisse du point
        @param y: ordonnée du point
        """
        self.x_ = x
        self.y_ = y


    def calcule_angle(self, ancre):
        """

        @param ancre:   point de référence plus bas et plus à gauche que tous les autres points
        @return:         l'angle entre l'horizontale passant par ancre
                        et le segment (ancre - self )
                        l'angle est en fait représenté par son cosinus
        """
        den = (self.x_ - ancre.x_) ** 2 + (self.y_ - ancre.y_) ** 2
        angle = (self.x_ - ancre.x_) / sqrt(den)

        return angle

    def getx(self):
        """ Accesseur pour x_

        @return:     attribut x_
        """
        return self.x_

    def gety(self):
        """ Accesseur pour y_

        @return:     attribut y_
        """
        return self.y_

    def __lt__(self, autre_point):
        """

        @param autre_point:     autre_point point
        @return:                 True si le point est plus en bas ou plus à gauche que le second point
        """

        if self.y_ < autre_point.y_:
            return True

        elif self.y_ == autre_point.y_:
            return self.x_ < autre_point.x_

        return False

def ccw (point1, point2, point3):
# *******************************

# fonction qui calcule si pour passer du point1, au point2
# puis au point3 on tourne dans le sens des aiguilles d'une montre (-1)
# ou dans le sens contraire des aiguilles d'une montre (1)
# rend 0 s'ils sont sur la meme ligne */
# ccw : counterclockwise */

    dx1 = point2.getx() - point1.getx() ;		#  dy/dx : pente de la droite
    dy1 = point2.gety() - point1.gety() ;
    dx2 = point3.getx() - point1.getx() ;
    dy2 = point3.gety() - point1.gety() ;

    if (dx1*dy2 > dy1*dx2):
        return +1
    if (dx1*dy2 < dy1*dx2):
        return -1
    if (dx1*dx2 < 0 or dy1*dy2 < 0):
        return -1
    if ((dx1*dx1 + dy1*dy1) < (dx2*dx2 + dy2*dy2)):
        return +1
    return 0


class Nuage:

    def __init__(self,nom_fichier):
        with open(nom_fichier, 'r') as file:
            self.ensemble_point_ = set()
            for ligne in file:
                ligne = ligne.split()
                self.ensemble_point_.add(Point(float(ligne[0]), float(ligne[1])))#creation objet type point et convertir chaine de caractere en nombre

    def calcul_ancre(self):
        return min(self.ensemble_point_)

    def calcul_polygone(self):
        """

        @return: un polygone a partir d'un ensemble de pts par la méthode :
                 - tri des pts par angle croissant selon l'angle avec l'horizontale
        """
        # le polygone est constitué d'une liste de tous les points du nuage
        polygone = list(self.ensemble_point_)
        # recherche du point le plus bas
        ancre = self.calcul_ancre()

        # on enleve ancre pour ne pas avoir de problème de division par zéro qd on calcule l'angle
        polygone.remove(ancre)

        # on trie la liste des points en fonction de l'angle
        # les angles varient dans l'ordre inverse des consinus
        # il faut donc faire un tri inverse
        polygone.sort(key=lambda pt: pt.calcule_angle(ancre), reverse=True)

        # on remet l'ancre dans le poloygone
        polygone.insert(0, ancre)

        return polygone

    def calcul_enveloppe_graham(self):
        """

        @return:     l'enveloppe convexe "entourant" le nuage de points
        """

        # au depart, on dispose du polygone constitué des points du nuage de points classés
        # par angle croissant par rapport à l'horizontale qui passe par l'ancre
        polygone = self.calcul_polygone()

        # on ajoute le premier point à la fin du polygone
        # pour que l'avant dernier point soit traité comme les autres
        polygone.append(polygone[0])

        #  Examen des points du polygone
        #  sauf les deuxs premiers points qui sont d'office sur l'enveloppe
        # et qu'on a mis au début du polygone

        enveloppe = [polygone[0], polygone[1]]
        prec = polygone[1]
        for i in range(2, len(polygone) - 1):
            # i represente le rang du point examine dans le polygone
            point = polygone[i]
            psuiv = polygone[i + 1]

            if ccw(prec, point, psuiv) != -1:
                # le point pt est sur l'enveloppe, on l'ajoute
                # et on avance
                enveloppe.append(point)
            else:
                # le point pt n'est pas sur l'enveloppe
                # on ne l'ajoute pas et on recule sur l'enveloppe
                point = prec
                prec = enveloppe[-2]

                while (ccw(prec, point, psuiv) == -1):
                    # La suppression de P entraine peut-etre d'autres suppressions
                    # sur les points precedents
                    # on boucle tant qu'on supprime des points
                    enveloppe.pop()
                    point = prec
                    prec = enveloppe[-2]

            prec = point
            # on réduit l'enveloppe convexe à seulement les points qui sont dans l'enveloppe
        return enveloppe


with open("test.dta", "w") as test:
    for i in range(100):
        test.write(str(randint(-100, 100)) + " " + str(randint(-100, 100)) + "\n")

# nom_fichier = input("Saisissez le nom du fichier de données (***.dta) : ")
nom_fichier = 'test.dta'
mon_nuage = Nuage(nom_fichier)
hull = mon_nuage.calcul_enveloppe_graham()


polygone = []
for i in range(len(hull)):
    pt = (hull[i].getx(), hull[i].gety())
    polygone.append(pt)

a = 0
b = 0
c = 0

vertexA = (0.0, 0.0)
vertexB = (0.0, 0.0)
vertexC = (0.0, 0.0)

bestArea = float("inf")
bestA = (0.0, 0.0)
bestB = (0.0, 0.0)
bestC = (0.0, 0.0)



def predecessor(point_index):
    return point_index - 1 if point_index != 0 else len(polygone) - 1


def successor(point_index):
    return (point_index + 1) % len(polygone)


def find_minimal_enclosing_triangle():
    global a, b, c
    for c in range(len(polygone)):
        for a in range(len(polygone)):
            if areValidIntersectingLines(predecessor(a), a, predecessor(c), c):
                for b in range(len(polygone)):
                    if areValidPolygonPoints(c, a, b):
                        findMinEnclosingTriangleTangentSideB()
                        findMinEnclosingTriangleFlushSideB()


def areValidIntersectingLines(index_a1, index_a2, index_b1, index_b2):
    a1 = polygone[index_a1]
    a2 = polygone[index_a2]
    b1 = polygone[index_b1]
    b2 = polygone[index_b2]

    s = np.vstack([a1, a2, b1, b2])  # s for stacked
    h = np.hstack((s, np.ones((4, 1))))  # h for homogeneous
    l1 = np.cross(h[0], h[1])  # get first line
    l2 = np.cross(h[2], h[3])  # get second line
    x, y, z = np.cross(l1, l2)  # point of intersection
    if z == 0:  # lines are parallel
        return False
    else:
        global vertexB
        vertexB = (x / z, y / z)
        return True


def areValidPolygonPoints(a, b, c):
    return (a != c) and (a != b) and (b != c)


def findMinEnclosingTriangleTangentSideB():
    if (b != predecessor(c)) and (b != predecessor(a)):
        sideCParameters = lineEquationParameters(polygone[predecessor(c)], polygone[c])
        sideAParameters = lineEquationParameters(polygone[predecessor(a)], polygone[a])

        if not areParallelLines(sideCParameters, sideAParameters):
            updateVerticesCAndA(sideCParameters, sideAParameters)

            if isValidTangentSideB():
                computeEnclosingTriangle()


def lineIntersection(a1, b1, a2, b2):
    A1 = b1[1] - a1[1]
    B1 = a1[0] - b1[0]
    C1 = (a1[0] * A1) + (a1[1] * B1)

    A2 = b2[1] - a2[1]
    B2 = a2[0] - b2[0]
    C2 = (a2[0] * A2) + (a2[1] * B2)

    det = (A1 * B2) - (A2 * B1)

    if not almostEqual(det, 0):
        intersection = (float(((C1 * B2) - (C2 * B1)) / (det)), float(((C2 * A1) - (C1 * A2)) / (det)))
        return intersection
    else:
        return False


def findMinEnclosingTriangleFlushSideB():
    # If vertices A and C exist
    global vertexA, vertexC
    vertexA = lineIntersection(polygone[predecessor(b)], polygone[b], polygone[predecessor(c)], polygone[c])
    vertexC = lineIntersection(polygone[predecessor(a)], polygone[a], polygone[predecessor(b)], polygone[b])

    if vertexA is not False and vertexC is not False:
        computeEnclosingTriangle()


def lineEquationParameters(point1, point2):  # line equation ax + by + c = 0
    param_a = point2[1] - point1[1]
    param_b = point1[0] - point2[0]
    param_c = ((-point1[1]) * param_b) - (point1[0] * param_a)
    return param_a, param_b, param_c


def areParallelLines(sideCParameters, sideAParameters):
    determinant = (sideCParameters[0] * sideAParameters[1]) - (sideAParameters[0] * sideCParameters[1])
    return almostEqual(determinant, 0)


def updateVerticesCAndA(sideCParameters, sideAParameters):
    # Side A parameters
    a1 = sideCParameters[0]
    b1 = sideCParameters[1]
    c1 = sideCParameters[2]

    # Side B parameters
    a2 = sideAParameters[0]
    b2 = sideAParameters[1]
    c2 = sideAParameters[2]

    # Polygon point "b" coordinates
    m = polygone[b][0]
    n = polygone[b][1]

    # Compute vertices A and C x-coordinates
    x2 = ((2 * b1 * b2 * n) + (c1 * b2) + (2 * a1 * b2 * m) + (b1 * c2)) / ((a1 * b2) - (a2 * b1))
    x1 = (2 * m) - x2

    # Compute vertices A and C y-coordinates
    y2 = 0.0
    y1 = 0.0

    if almostEqual(b1, 0):  # b1 = 0 and b2 != 0
        y2 = ((-c2) - (a2 * x2)) / b2
        y1 = (2 * n) - y2
    elif almostEqual(b2, 0):  # b1 != 0 and b2 = 0
        y1 = ((-c1) - (a1 * x1)) / b1
        y2 = (2 * n) - y1
    else:  # b1 != 0 and b2 != 0
        y1 = ((-c1) - (a1 * x1)) / b1
        y2 = ((-c2) - (a2 * x2)) / b2

    # Update vertices A and C coordinates
    global vertexA, vertexC
    vertexA = (x1, y1)
    vertexC = (x2, y2)


def angleOfLineWrtOxAxis(point1, point2):
    y = point2[1] - point1[1]
    x = point2[0] - point1[0]

    angle = (atan2(y, x) * 180 / 3.141592)

    return angle + 360 if angle < 0 else angle


def almostEqual(val1, val2):
    return abs(val1 - val2) < 0.0001


def isAngleBetween(angle1, angle2, angle3):
    if (int(angle2 - angle3) % 180) > 0:
        return (angle3 < angle1) and (angle1 < angle2)
    else:
        return (angle2 < angle1) and (angle1 < angle3)


def lessOrEqual(val1, val2):
    return (val1 < val2) or (almostEqual(val1, val2))


def isAngleBetweenNonReflex(angle1, angle2, angle3):
    if abs(angle2 - angle3) > 180:
        if angle2 > angle3:
            return ((angle2 < angle1) and lessOrEqual(angle1, 360)) or (lessOrEqual(0, angle1) and (angle1 < angle3))
        else:
            return ((angle3 < angle1) and lessOrEqual(angle1, 360)) or (lessOrEqual(0, angle1) and (angle1 < angle2))
    else:
        return isAngleBetween(angle1, angle2, angle3)


def oppositeAngle(angle):
    return angle - 180 if angle > 180 else angle + 180


def isOppositeAngleBetweenNonReflex(angle1, angle2, angle3):
    angle1Opposite = oppositeAngle(angle1)

    return isAngleBetweenNonReflex(angle1Opposite, angle2, angle3)


def isValidTangentSideB():
    angleOfTangentSideB = angleOfLineWrtOxAxis(vertexC, vertexA)
    anglePredecessor = angleOfLineWrtOxAxis(polygone[predecessor(b)], polygone[b])
    angleSuccessor = angleOfLineWrtOxAxis(polygone[b], polygone[successor(b)])

    return (isAngleBetweenNonReflex(angleOfTangentSideB, anglePredecessor, angleSuccessor)) or (
        isOppositeAngleBetweenNonReflex(angleOfTangentSideB, anglePredecessor, angleSuccessor))


def ptInTriang(p_test, p0, p1, p2):
    dX = p_test[0] - p0[0]
    dY = p_test[1] - p0[1]
    dX20 = p2[0] - p0[0]
    dY20 = p2[1] - p0[1]
    dX10 = p1[0] - p0[0]
    dY10 = p1[1] - p0[1]

    s_p = (dY20 * dX) - (dX20 * dY)
    t_p = (dX10 * dY) - (dY10 * dX)
    D = (dX10 * dY20) - (dY10 * dX20)

    if D > 0:
        return (s_p >= 0) and (t_p >= 0) and (s_p + t_p) <= D
    else:
        return (s_p <= 0) and (t_p <= 0) and (s_p + t_p) >= D


def isValidMinimalTriangle():
    for i in range(len(polygone)):
        if not ptInTriang(polygone[i], vertexA, vertexB, vertexC):
            return False
    return True


def updateMinEnclosingTriangle():
    global bestArea, bestA, bestB, bestC
    area = abs(0.5 * (vertexA[0] * (vertexB[1] - vertexC[1]) + vertexB[0] * (vertexC[1] - vertexA[1]) + vertexC[0] * (vertexA[1] - vertexB[1])))
    if area < bestArea:
        bestArea = area
        bestA = vertexA
        bestB = vertexB
        bestC = vertexC


def computeEnclosingTriangle():
    if isValidMinimalTriangle():
        updateMinEnclosingTriangle()
    return




def show_points_polygone_and_triangle():

    # Points

    plt.figure()
    with open(nom_fichier, 'r') as file:
        for ligne in file:
            ligne = ligne.split()
            px = float(ligne[0])
            py = float(ligne[1])
            plt.plot(px, py, 'bo')

    # polygone

    # segments = []
    # for i in range(-1, len(polygone) - 1):
    #     pt1 = (polygone[i][0], polygone[i][1])
    #     pt2 = (polygone[i + 1][0], polygone[i + 1][1])
    #
    #     segments.append((pt1, pt2))
    #
    # for (beginning, end) in segments:
    #     plt.plot([beginning[0], end[0]], [beginning[1], end[1]])

    # Triangle

    plt.plot([bestA[0], bestB[0]], [bestA[1], bestB[1]])
    plt.plot([bestB[0], bestC[0]], [bestB[1], bestC[1]])
    plt.plot([bestC[0], bestA[0]], [bestC[1], bestA[1]])

    plt.suptitle(legende)

    plt.show()


find_minimal_enclosing_triangle()

legende = "Les sommets du triangle sont : " + "\n" \
  + str(bestA[0])[:6] + ", " + str(bestA[1])[:6] + "\n" \
  + str(bestB[0])[:6] + ", " + str(bestB[1])[:6] + "\n" \
  + str(bestC[0])[:6] + ", " + str(bestC[1])[:6]

show_points_polygone_and_triangle()
print(legende)