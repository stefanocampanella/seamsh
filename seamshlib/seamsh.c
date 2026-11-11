/*
 * seamsh - Copyright (C) <2010-2020>
 * <Universite catholique de Louvain (UCL), Belgium
 *
 * List of the contributors to the development of seamsh: see AUTHORS file.
 * Description and complete License: see LICENSE file.
 *
 * This program (seamsh) is free software:
 * you can redistribute it and/or modify it under the terms of the GNU Lesser General
 * Public License as published by the Free Software Foundation, either version
 * 3 of the License, or (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU Lesser General Public License for more details.
 *
 * You should have received a copy of the GNU Lesser General Public License
 * along with this program (see COPYING file).  If not,
 * see <http://www.gnu.org/licenses/>.
 */

/*
 * Boundary coarsening.
 *
 * Input: points sampled on the boundary curves of a domain (about two per target size), each with
 * the tags of its curves and the target size there, a Delaunay triangulation of these points, and
 * "seed" triangles known to be in the water. Output: the boundary of the water reachable from the
 * seeds, as closed loops of points about one target size apart, cut into tagged curves. Seeds in
 * waters that are not connected (at this resolution) give a domain made of several regions.
 *
 * Steps (gen_boundaries_from_points):
 *  1. color_triangles: flood fill from the seed triangles. The fill goes from a triangle to its
 *     neighbour only through an edge longer than the target size. Two points on opposite shores of
 *     a channel narrower than the size are joined by a shorter edge, so the fill does not enter
 *     such a channel, and consecutive samples of one curve are closer than the size, so it does not
 *     cross the coast. Colored triangles are the domain.
 *  2. optimize_mesh: edge collapses. Points that touch no colored triangle are removed, and edges
 *     of the domain boundary shorter than 0.7 times the size are collapsed, which brings the
 *     boundary to the target resolution.
 *  3. extract_boundaries: the edges between colored and other triangles are chained into closed
 *     loops, cut where the curve tag changes, and moved a little towards the domain.
 *
 * Data layout: `tri` holds 3 point indices per triangle, counter-clockwise; a removed triangle has
 * tri[3 * t] == -1. `neigh[3 * t + j]` is the triangle on the other side of edge j of t, which
 * joins its corners j and (j + 1) % 3, or -1. `x` holds 2 coordinates per point.
 */

#include <math.h>
#include <setjmp.h>
#include <stdlib.h>

/* Return values of gen_boundaries_from_points. */
enum { SEAMSH_OK = 0, SEAMSH_OUT_OF_MEMORY = 1, SEAMSH_BAD_TRIANGULATION = 2 };

#if defined(_MSC_VER)
#define THREAD_LOCAL __declspec(thread)
#else
#define THREAD_LOCAL __thread
#endif

/*
 * Errors. The library is loaded in a Python process, so it must never exit: an error jumps back to
 * gen_boundaries_from_points, which frees the memory allocated so far and returns an error code.
 * To make that possible, every allocation of a call is recorded in `owned`. The state is per
 * thread, since ctypes releases the Python lock during the call.
 */
#define MAX_OWNED 32
static THREAD_LOCAL jmp_buf error_jump;
static THREAD_LOCAL void *owned[MAX_OWNED];
static THREAD_LOCAL int n_owned;
/* Upper bound of the number of steps of any walk over triangles: a longer one means that the
 * triangulation is inconsistent, and would otherwise never end. */
static THREAD_LOCAL long max_steps;

static void fail(int error) { longjmp(error_jump, error); }

static void count_step(long *steps) {
  if (++*steps > max_steps)
    fail(SEAMSH_BAD_TRIANGULATION);
}

static void *owned_malloc(size_t size) {
  void *p = malloc(size > 0 ? size : 1);
  if (p == NULL || n_owned == MAX_OWNED) {
    free(p);
    fail(SEAMSH_OUT_OF_MEMORY);
  }
  owned[n_owned++] = p;
  return p;
}

static int owned_index(const void *p) {
  for (int i = 0; i < n_owned; ++i) {
    if (owned[i] == p)
      return i;
  }
  fail(SEAMSH_OUT_OF_MEMORY);
  return -1;
}

static void *owned_realloc(void *old, size_t size) {
  int i = owned_index(old);
  void *p = realloc(old, size);
  if (p == NULL)
    fail(SEAMSH_OUT_OF_MEMORY);
  owned[i] = p;
  return p;
}

/* Forgets an allocation without freeing it (it is handed over to the caller). */
static void owned_release(const void *p) {
  if (p != NULL) {
    int i = owned_index(p);
    owned[i] = owned[--n_owned];
  }
}

static void owned_free(void *p) {
  if (p != NULL) {
    owned_release(p);
    free(p);
  }
}

static void owned_free_all(void) {
  while (n_owned > 0)
    free(owned[--n_owned]);
}

/* Appends one element to a growing array of `*size` elements and returns its address. The capacity
 * is the smallest power of two that holds the elements (at least 2). */
static void *vector_append(void **p, int *size, int elem_size) {
  if (*size == 0) {
    *p = owned_malloc((size_t)elem_size * 2);
  } else if (*size >= 2 && (*size & (*size - 1)) == 0) {
    *p = owned_realloc(*p, (size_t)(*size) * 2 * elem_size);
  }
  *size += 1;
  return ((char *)*p) + ((size_t)(*size - 1) * elem_size);
}

static int coord_vector_append(double **points, int *n_points, const double *x) {
  double *xn = (double *)vector_append((void **)points, n_points, 2 * sizeof(double));
  xn[0] = x[0];
  xn[1] = x[1];
  return (*n_points) - 1;
}

static int int_vector_append(int **v, int *n, int i) {
  return *(int *)vector_append((void **)v, n, sizeof(int)) = i;
}

static double double_vector_append(double **v, int *n, double i) {
  return *(double *)vector_append((void **)v, n, sizeof(double)) = i;
}

static double triangle_signed_area(const double *a, const double *b, const double *c) {
  double dxdxi[2][2] = {{b[0] - a[0], c[0] - a[0]}, {b[1] - a[1], c[1] - a[1]}};
  double det = dxdxi[0][0] * dxdxi[1][1] - dxdxi[1][0] * dxdxi[0][1];
  return det / 2;
}

static double vdist(const double *v0, const double *v1) {
  return hypot(v1[0] - v0[0], v1[1] - v0[1]);
}

/* Position (0, 1 or 2) of i among the three values of t. */
static int getid3(const int t[3], int i) {
  if (t[0] == i)
    return 0;
  if (t[1] == i)
    return 1;
  if (t[2] == i)
    return 2;
  fail(SEAMSH_BAD_TRIANGULATION);
  return -1;
}

/*
 * Walks around a point.
 *
 * The triangles around point v are visited from triangle t by crossing the edge of v that follows
 * it, until the walk is back in t or reaches the border of the triangulation; in that case the
 * other triangles are visited from t in the other direction, crossing the edge that precedes v.
 * All the loops below follow this pattern.
 */

/* Whether one of the triangles around v (starting from t) has corner w. */
static int fan_has_vertex(int t, int v, int w, const int *tri, const int *neigh) {
  long steps = 0;
  int cur = t;
  do {
    count_step(&steps);
    int iv = getid3(&tri[cur * 3], v);
    if (tri[cur * 3 + (iv + 1) % 3] == w || tri[cur * 3 + (iv + 2) % 3] == w)
      return 1;
    cur = neigh[cur * 3 + iv];
  } while (cur != t && cur != -1);
  if (cur == -1) {
    cur = t;
    do {
      count_step(&steps);
      int iv = getid3(&tri[cur * 3], v);
      if (tri[cur * 3 + (iv + 1) % 3] == w || tri[cur * 3 + (iv + 2) % 3] == w)
        return 1;
      cur = neigh[cur * 3 + (iv + 2) % 3];
    } while (cur != t && cur != -1);
  }
  return 0;
}

/* Whether v is on the border of the triangulation (the triangles around it do not close). */
static int fan_is_open(int t, int v, const int *tri, const int *neigh) {
  long steps = 0;
  int cur = t;
  do {
    count_step(&steps);
    cur = neigh[cur * 3 + getid3(&tri[cur * 3], v)];
  } while (cur != t && cur != -1);
  return cur == -1;
}

/* Number of points joined by an edge to both v0 and v1, two corners of t. */
static int count_common_neighbours(int t, int v0, int v1, const int *tri, const int *neigh) {
  long steps = 0;
  int count = 0;
  int cur = t;
  int backward = 0;
  for (;;) {
    count_step(&steps);
    int iv = getid3(&tri[cur * 3], v0);
    /* Each neighbour of v0 is counted once: as the last corner of the triangle that precedes
     * their edge, or as the second corner when that edge is on the border. */
    int a = tri[cur * 3 + (iv + 1) % 3];
    int b = tri[cur * 3 + (iv + 2) % 3];
    if (b != v1 && fan_has_vertex(t, v1, b, tri, neigh))
      count++;
    if (neigh[cur * 3 + iv] == -1 && a != v1 && fan_has_vertex(t, v1, a, tri, neigh))
      count++;
    int next = neigh[cur * 3 + (backward ? (iv + 2) % 3 : iv)];
    if (next == -1 && !backward) {
      backward = 1;
      next = neigh[t * 3 + (getid3(&tri[t * 3], v0) + 2) % 3];
    }
    if (next == -1 || next == t)
      return count;
    cur = next;
  }
}

/*
 * On boundary edge e0 of colored triangle t0, joining v0 and v1: where to put the point that
 * replaces them, as the weight of v0 (1: on v0, 0: on v1). With d the length of the edge, d0 and d1
 * those of the boundary edges before v0 and after v1, the point is put where it divides the length
 * d0 + d + d1 in two halves.
 */
static double optimal_boundary_collapse(int t0, int e0, const int *tri, const int *neigh,
                                        const double *x, const int *tri_color) {
  int v0 = tri[t0 * 3 + e0];
  int v1 = tri[t0 * 3 + (e0 + 1) % 3];
  double d = vdist(&x[v0 * 2], &x[v1 * 2]);
  double d1 = 0, d0 = 0;
  long steps = 0;
  int cur = t0;
  do {
    count_step(&steps);
    int iv1 = getid3(&tri[cur * 3], v1);
    int n = neigh[cur * 3 + iv1];
    if (n == -1 || tri_color[n] != tri_color[cur]) {
      d1 = vdist(&x[v1 * 2], &x[tri[cur * 3 + (iv1 + 1) % 3] * 2]);
      break;
    }
    cur = n;
  } while (cur != t0);
  cur = t0;
  do {
    count_step(&steps);
    int iv0 = getid3(&tri[cur * 3], v0);
    int n = neigh[cur * 3 + (iv0 + 2) % 3];
    if (n == -1 || tri_color[n] != tri_color[cur]) {
      d0 = vdist(&x[v0 * 2], &x[tri[cur * 3 + (iv0 + 2) % 3] * 2]);
      break;
    }
    cur = n;
  } while (cur != t0);
  double xi = fmin(1, fmax(0, (0.5 * (d + d0 + d1) - d0) / d));
  return 1 - xi;
}

/* Replaces corner v0 by v1 in the triangles around v0, except in t0. */
static void substitute_vertex(int t0, int v0, int v1, int *tri, const int *neigh) {
  long steps = 0;
  int cur = t0;
  do {
    count_step(&steps);
    int iv0 = getid3(&tri[cur * 3], v0);
    if (cur != t0)
      tri[cur * 3 + iv0] = v1;
    cur = neigh[cur * 3 + iv0];
  } while (cur != t0 && cur != -1);
  if (cur == -1) {
    cur = t0;
    do {
      count_step(&steps);
      int iv0 = getid3(&tri[cur * 3], v0);
      if (cur != t0)
        tri[cur * 3 + iv0] = v1;
      cur = neigh[cur * 3 + (iv0 + 2) % 3];
    } while (cur != t0 && cur != -1);
  }
}

/* Makes t0 and t1, the two other neighbours of the removed triangle `removed`, neighbours of each
 * other. */
static void substitute_neighbour(int t0, int t1, int removed, int *neigh) {
  if (t0 >= 0)
    neigh[t0 * 3 + getid3(&neigh[t0 * 3], removed)] = t1;
  if (t1 >= 0)
    neigh[t1 * 3 + getid3(&neigh[t1 * 3], removed)] = t0;
}

/* Whether moving point v to `moveto` keeps every triangle around it counter-clockwise, leaving
 * aside triangles ignore0 and ignore1 (those that the collapse removes). */
static int check_vertex_move(int t, int v, const double *moveto, int ignore0, int ignore1,
                             const int *tri, const int *neigh, const double *x) {
  long steps = 0;
  int cur = t;
  do {
    count_step(&steps);
    int iv = getid3(&tri[cur * 3], v);
    if (cur != ignore0 && cur != ignore1) {
      if (triangle_signed_area(moveto, &x[tri[cur * 3 + (iv + 1) % 3] * 2],
                               &x[tri[cur * 3 + (iv + 2) % 3] * 2]) < 0)
        return 0;
    }
    cur = neigh[cur * 3 + iv];
  } while (cur != t && cur != -1);
  if (cur == -1) {
    cur = t;
    do {
      count_step(&steps);
      int iv = getid3(&tri[cur * 3], v);
      if (cur != ignore0 && cur != ignore1 && cur != t) {
        if (triangle_signed_area(moveto, &x[tri[cur * 3 + (iv + 1) % 3] * 2],
                                 &x[tri[cur * 3 + (iv + 2) % 3] * 2]) < 0)
          return 0;
      }
      cur = neigh[cur * 3 + (iv + 2) % 3];
    } while (cur != t && cur != -1);
  }
  return 1;
}

/*
 * Tags of the points. `tag` holds, for each point, a linked list of the tags of the curves it
 * belongs to: entry i is (tag, index of the next entry or -1), and the list of point v starts at
 * entry v. Entries beyond the points are the further tags of points on several curves.
 */

/* Gives v0 the tags of v1 too: the list of v1 is attached at the end of the list of v0. */
static void merge_tag_list(int *tag, int v0, int v1) {
  int last = v0;
  while (tag[last * 2 + 1] != -1)
    last = tag[last * 2 + 1];
  tag[last * 2 + 1] = v1;
}

static int has_tag(const int *tag, int v, int t) {
  while (v != -1) {
    if (tag[v * 2] == t)
      return 1;
    v = tag[v * 2 + 1];
  }
  return 0;
}

/* The tag of the boundary segment from v0 to v1: the current tag if both points have it, else a
 * tag that both have, else the current tag if one of them has it, else the first tag of v0. */
static int get_segment_tag(const int *tag, int v0, int v1, int current_tag) {
  if (has_tag(tag, v0, current_tag) && has_tag(tag, v1, current_tag))
    return current_tag;
  for (int i = v0; i != -1; i = tag[i * 2 + 1]) {
    if (has_tag(tag, v1, tag[i * 2]))
      return tag[i * 2];
  }
  if (has_tag(tag, v0, current_tag) || has_tag(tag, v1, current_tag))
    return current_tag;
  return tag[v0 * 2];
}

/*
 * Collapses edge e0 of triangle t0, which joins v0 and v1: both points are replaced by one point
 * at xi * v0 + (1 - xi) * v1 (stored as v0), and the triangles on both sides of the edge (t0 and
 * t1, if any) are removed. Returns 0, leaving everything as it was, if the collapse is not valid.
 *
 *      v1
 *      /|\
 *     / | \
 *    /  |  \             v0
 *   /   |   \           _/\_
 *  / t0 | t1 \        _/    \_
 * /_____|_____\      /        \
 *      v0
 */
static int collapse_edge(int t0, int e0, double xi, int *tri, int *neigh, double *x, double *length,
                         int *vcolor, int *tag) {
  int v0 = tri[t0 * 3 + e0];
  int v1 = tri[t0 * 3 + (e0 + 1) % 3];
  double a = xi, b = 1 - a;
  int t1 = neigh[t0 * 3 + e0];
  double xmid[2] = {a * x[v0 * 2 + 0] + b * x[v1 * 2 + 0], a * x[v0 * 2 + 1] + b * x[v1 * 2 + 1]};
  /* No triangle may be turned over. */
  if (!check_vertex_move(t0, v1, xmid, t0, t1, tri, neigh, x))
    return 0;
  if (!check_vertex_move(t0, v0, xmid, t0, t1, tri, neigh, x))
    return 0;
  /* The triangulation must stay a valid one ("link condition"): the only points joined to both v0
   * and v1 must be the third corners of the removed triangles, otherwise two triangles would end
   * up on the same three points or sharing two edges; and an inner edge between two points of the
   * border would pinch the triangulation. Without this check, later walks around such points
   * never end. */
  if (count_common_neighbours(t0, v0, v1, tri, neigh) != (t1 >= 0 ? 2 : 1))
    return 0;
  if (t1 >= 0 && fan_is_open(t0, v0, tri, neigh) && fan_is_open(t0, v1, tri, neigh))
    return 0;
  x[v0 * 2 + 0] = xmid[0];
  x[v0 * 2 + 1] = xmid[1];
  length[v0] = a * length[v0] + b * length[v1];
  vcolor[v0] = vcolor[v0] > vcolor[v1] ? vcolor[v0] : vcolor[v1];
  merge_tag_list(tag, v0, v1);
  substitute_vertex(t0, v1, v0, tri, neigh);
  substitute_neighbour(neigh[t0 * 3 + (e0 + 1) % 3], neigh[t0 * 3 + (e0 + 2) % 3], t0, neigh);
  tri[t0 * 3] = -1;
  if (t1 >= 0) {
    int e1 = getid3(&neigh[t1 * 3], t0);
    substitute_neighbour(neigh[t1 * 3 + (e1 + 1) % 3], neigh[t1 * 3 + (e1 + 2) % 3], t1, neigh);
    tri[t1 * 3] = -1;
  }
  return 1;
}

/* Orders half-edges (4 ints: smaller point, larger point, triangle, edge) by their points. */
static int half_edge_compare(const void *p0, const void *p1) {
  const int *i0 = (const int *)p0;
  const int *i1 = (const int *)p1;
  return i0[0] == i1[0] ? i0[1] - i1[1] : i0[0] - i1[0];
}

/* The neighbours of the triangles (see the data layout above): the edges of all triangles are
 * sorted by their points, which puts the two sides of an edge next to each other. */
static int *build_neighbours(int n_tri, const int *tri) {
  int *half_edges = owned_malloc(sizeof(int) * 4 * n_tri * 3);
  int n_half_edges = 0;
  for (int i = 0; i < n_tri; ++i) {
    const int *t = tri + i * 3;
    if (t[0] < 0)
      continue;
    for (int j = 0; j < 3; ++j) {
      int *he = half_edges + n_half_edges * 4;
      int j1 = (j + 1) % 3;
      if (t[j1] < t[j]) {
        he[0] = t[j1];
        he[1] = t[j];
      } else {
        he[0] = t[j];
        he[1] = t[j1];
      }
      he[2] = i;
      he[3] = j;
      n_half_edges++;
    }
  }
  qsort(half_edges, n_half_edges, 4 * sizeof(int), half_edge_compare);
  /* An edge with more than two triangles: not a triangulation of the plane. */
  for (int i = 0; i + 2 < n_half_edges; ++i) {
    if (half_edge_compare(half_edges + i * 4, half_edges + i * 4 + 4) == 0 &&
        half_edge_compare(half_edges + i * 4, half_edges + i * 4 + 8) == 0)
      fail(SEAMSH_BAD_TRIANGULATION);
  }
  int *neighbours = owned_malloc(sizeof(int) * n_tri * 3);
  for (int i = 0; i < n_tri * 3; ++i) {
    neighbours[i] = -1;
  }
  for (int i = 0; i + 1 < n_half_edges; ++i) {
    int *he0 = half_edges + i * 4;
    int *he1 = half_edges + i * 4 + 4;
    if (half_edge_compare(he0, he1) == 0) {
      neighbours[he0[2] * 3 + he0[3]] = he1[2];
      neighbours[he1[2] * 3 + he1[3]] = he0[2];
      i++;
    }
  }
  owned_free(half_edges);
  return neighbours;
}

/*
 * Step 3: the boundary of the colored triangles.
 *
 * Output: `points` (n_points x 2 coordinates) and `lines`, a sequence of curves, each one written
 * as: its tag, the indices of its points in `points`, -1. The curves of a loop follow each other,
 * each starting on the last point of the one before, and the last one ends on the first point of
 * the loop.
 */
static void extract_boundaries(const double *x, const double *size, const int *tag, int n_tri,
                               const int *tri, const int *tri_color, double **points, int *n_points,
                               int **lines, int *n_lines) {
  /* Keep the colored triangles only: their edges without a neighbour are the boundary. */
  int n_kept = 0;
  int *kept = owned_malloc(sizeof(int) * n_tri * 3);
  for (int i = 0; i < n_tri; ++i) {
    if (tri[i * 3] < 0 || tri_color[i] != 1)
      continue;
    for (int j = 0; j < 3; ++j) {
      kept[n_kept * 3 + j] = tri[i * 3 + j];
    }
    n_kept++;
  }
  int *neighbours = build_neighbours(n_kept, kept);
  int *touched = owned_malloc(sizeof(int) * n_kept);
  for (int i = 0; i < n_kept; ++i)
    touched[i] = 0;

  /* Chain the boundary edges into closed loops: sequences of point indices, each ended by -1. From
   * a boundary edge, the next one is found by turning around its end point through the triangles
   * until an edge without a neighbour. Triangles with a boundary edge are marked on the way, so
   * that each loop is walked once. */
  int *loops = NULL;
  int n_loops = 0;
  long steps = 0;
  for (int i = 0; i < n_kept; ++i) {
    if (touched[i] == 1)
      continue;
    for (int j = 0; j < 3; ++j) {
      if (neighbours[i * 3 + j] >= 0)
        continue;
      int first_point = kept[i * 3 + j];
      int t = i;
      int p = j;
      touched[i] = 1;
      int point;
      do {
        point = kept[t * 3 + (p + 1) % 3];
        int_vector_append(&loops, &n_loops, point);
        p = getid3(kept + t * 3, point);
        while (neighbours[t * 3 + p] >= 0) {
          count_step(&steps);
          t = neighbours[t * 3 + p];
          p = getid3(kept + t * 3, point);
        }
        touched[t] = 1;
        count_step(&steps);
      } while (point != first_point || t != i);
      int_vector_append(&loops, &n_loops, -1);
      break;
    }
  }

  double *point_size = NULL;
  int n_point_size = 0;
  for (int i = 0; i < n_loops; ++i) {
    const int *loop = loops + i;
    int length = 0;
    for (length = 0; loop[length] != -1; ++length)
      ;
    /* Start the loop where the tag changes (if it does), so that no curve spans the start. */
    int start = 0;
    int current_tag = get_segment_tag(tag, loop[0], loop[1], -1);
    for (int j = 1; j < length; ++j) {
      int next_tag = get_segment_tag(tag, loop[j], loop[(j + 1) % length], current_tag);
      if (current_tag != next_tag) {
        start = j;
        break;
      }
    }
    int first_point = *n_points;
    current_tag = get_segment_tag(tag, loop[start], loop[(start + 1) % length], -1);
    int_vector_append(lines, n_lines, current_tag);
    for (int j = 0; j < length; ++j) {
      int p = loop[(start + j) % length];
      int p_next = loop[(start + j + 1) % length];
      int segment_tag = get_segment_tag(tag, p, p_next, current_tag);
      coord_vector_append(points, n_points, x + p * 2);
      double_vector_append(&point_size, &n_point_size, size[p]);
      int_vector_append(lines, n_lines, (*n_points) - 1);
      if (segment_tag != current_tag) {
        /* The tag changes: end this curve on this point and start the next one from it. */
        int_vector_append(lines, n_lines, -1);
        int_vector_append(lines, n_lines, segment_tag);
        int_vector_append(lines, n_lines, (*n_points) - 1);
        current_tag = segment_tag;
      }
    }
    int_vector_append(lines, n_lines, first_point);
    int_vector_append(lines, n_lines, -1);
    i += length;

    /* Move the boundary a little towards the domain: each point by size / 50 along the sum of the
     * unit normals of the two segments that meet on it (the domain is on the left of the loop, and
     * so are the normals). The boundary then lies strictly inside the water of the original
     * curves, so the splines drawn through its points do not cross them. normals[j] is the normal
     * of the segment that ends on point j. */
    double *normals = owned_malloc(sizeof(double) * 2 * length);
    for (int j = 0; j < length; ++j) {
      const double *x0 = (*points) + (first_point + j) * 2;
      const double *x1 = (*points) + (first_point + (j + length - 1) % length) * 2;
      double *n = normals + j * 2;
      n[0] = x1[1] - x0[1];
      n[1] = -x1[0] + x0[0];
      double norm = hypot(n[0], n[1]);
      if (norm > 0) {
        n[0] /= norm;
        n[1] /= norm;
      }
    }
    for (int j = 0; j < length; ++j) {
      const double *n0 = &normals[j * 2];
      const double *n1 = &normals[((j + 1) % length) * 2];
      (*points)[(first_point + j) * 2 + 0] += point_size[first_point + j] * (n0[0] + n1[0]) / (50);
      (*points)[(first_point + j) * 2 + 1] += point_size[first_point + j] * (n0[1] + n1[1]) / (50);
    }
    owned_free(normals);
  }
  owned_free(point_size);
  owned_free(loops);
  owned_free(neighbours);
  owned_free(touched);
  owned_free(kept);
}

/* Turns triangle t over: swaps its first two corners, and the neighbours of its last two edges. */
static void flip_triangle(int t, int *tri, int *neigh) {
  int corner = tri[t * 3];
  tri[t * 3] = tri[t * 3 + 1];
  tri[t * 3 + 1] = corner;
  int neighbour = neigh[t * 3 + 1];
  neigh[t * 3 + 1] = neigh[t * 3 + 2];
  neigh[t * 3 + 2] = neighbour;
}

/*
 * Gives all triangles the same orientation, counter-clockwise.
 *
 * The orientation is spread from triangle to neighbour (two neighbours agree when they go through
 * their common edge in opposite directions), then each connected set of triangles is turned over
 * as a whole if its total area is negative. In a valid triangulation this makes every triangle
 * counter-clockwise. Orienting each triangle by the sign of its own area would not do: the
 * triangulation of points sampled on a straight line holds triangles of almost no area, some of
 * them inside out, and giving those the opposite orientation of their neighbours breaks the walks
 * around points, which then never end.
 */
static void orient_triangles(const double *x, int n_tri, int *tri, int *neigh) {
  int *visited = owned_malloc(sizeof(int) * n_tri);
  int *queue = owned_malloc(sizeof(int) * n_tri);
  for (int i = 0; i < n_tri; ++i)
    visited[i] = 0;
  int tail = 0;
  for (int first = 0; first < n_tri; ++first) {
    if (visited[first])
      continue;
    int head = tail;
    int component = tail;
    double area = 0;
    visited[first] = 1;
    queue[tail++] = first;
    while (head < tail) {
      int c = queue[head++];
      const int *t = tri + c * 3;
      area += triangle_signed_area(x + t[0] * 2, x + t[1] * 2, x + t[2] * 2);
      for (int j = 0; j < 3; ++j) {
        int n = neigh[c * 3 + j];
        if (n < 0 || visited[n])
          continue;
        /* Edge j of c goes from a to b: n must go from b to a. */
        int a = t[j], b = t[(j + 1) % 3];
        if (tri[n * 3 + (getid3(&tri[n * 3], a) + 1) % 3] == b)
          flip_triangle(n, tri, neigh);
        visited[n] = 1;
        queue[tail++] = n;
      }
    }
    if (area < 0) {
      for (int i = component; i < tail; ++i)
        flip_triangle(queue[i], tri, neigh);
    }
  }
  owned_free(queue);
  owned_free(visited);
}

/* Step 1: flood fill from the seed triangles through the edges longer than the smaller target size
 * (`length`) of their two points. Returns the color of each triangle: 1 if reached, else 0. */
static int *color_triangles(const double *x, int n_tri, const int *tri, const int *neigh,
                            int n_seeds, const int *seeds, const double *length) {
  int *tri_color = owned_malloc(sizeof(int) * n_tri);
  for (int i = 0; i < n_tri; ++i) {
    tri_color[i] = 0;
  }
  int *stack = owned_malloc(sizeof(int) * n_tri);
  for (int s = 0; s < n_seeds; ++s) {
    /* A seed in a region that an earlier seed has filled adds nothing. */
    if (tri_color[seeds[s]])
      continue;
    int stack_p = 0;
    stack[0] = seeds[s];
    tri_color[seeds[s]] = 1;
    while (stack_p >= 0) {
      int c = stack[stack_p--];
      for (int j = 0; j < 3; ++j) {
        int n = neigh[c * 3 + j];
        if (n >= 0 && !tri_color[n]) {
          int i0 = tri[c * 3 + j];
          int i1 = tri[c * 3 + (j + 1) % 3];
          const double *p0 = x + 2 * i0;
          const double *p1 = x + 2 * i1;
          double d2 = (p1[0] - p0[0]) * (p1[0] - p0[0]) + (p1[1] - p0[1]) * (p1[1] - p0[1]);
          double lmin = fmin(length[i0], length[i1]);
          if (d2 > lmin * lmin) {
            tri_color[n] = 1;
            stack[++stack_p] = n;
          }
        }
      }
    }
  }
  owned_free(stack);
  return tri_color;
}

/*
 * Step 2: edge collapses, repeated over all triangles until nothing changes, or MAX_SWEEPS sweeps.
 * Returns 1 if it stopped because nothing changed, 0 if it was cut short (the boundary may then be
 * finer than asked in places).
 *  - Outside the domain (an edge between two triangles that are not colored, or on the border): a
 *    point of the edge that touches no colored triangle is removed by collapsing the edge onto its
 *    other point. What is left outside are triangles between boundary points only.
 *  - On the boundary (an edge of a colored triangle without colored neighbour): an edge shorter
 *    than 0.7 times the target size is collapsed, onto the point given by
 *    optimal_boundary_collapse or, if that is not valid, onto one of its ends.
 */
#define MAX_SWEEPS 100
static int optimize_mesh(double *x, double *length, int n_points, int n_tri, int *tri,
                         const int *tri_color, int *neigh, int *tag) {
  /* 1 for the points of colored triangles. */
  int *vcolor = owned_malloc(sizeof(int) * n_points);
  for (int i = 0; i < n_points; ++i) {
    vcolor[i] = 0;
  }
  for (int i = 0; i < n_tri; ++i) {
    if (tri_color[i] == 1) {
      for (int j = 0; j < 3; ++j) {
        vcolor[tri[i * 3 + j]] = 1;
      }
    }
  }

  size_t n_collapses;
  int sweep = 0;
  do {
    sweep += 1;
    n_collapses = 0;
    for (int i = 0; i < n_tri; i++) {
      int *t = tri + i * 3;
      for (int j = 0; j < 3 && t[0] != -1; ++j) {
        int n = neigh[i * 3 + j];
        int i0 = t[j];
        int i1 = t[(j + 1) % 3];
        if (tri_color[i] == 0 && (n == -1 || tri_color[n] != 1)) {
          if (vcolor[i1] == 0) {
            n_collapses += collapse_edge(i, j, 1, tri, neigh, x, length, vcolor, tag);
          } else if (vcolor[i0] == 0) {
            n_collapses += collapse_edge(i, j, 0, tri, neigh, x, length, vcolor, tag);
          }
        }
        if (tri_color[i] == 1 && (n == -1 || tri_color[n] == 0) &&
            vdist(&x[i0 * 2], &x[i1 * 2]) < 0.7 * fmin(length[i0], length[i1])) {
          double xi = optimal_boundary_collapse(i, j, tri, neigh, x, tri_color);
          n_collapses += collapse_edge(i, j, xi, tri, neigh, x, length, vcolor, tag) ||
                         collapse_edge(i, j, 0, tri, neigh, x, length, vcolor, tag) ||
                         collapse_edge(i, j, 1, tri, neigh, x, length, vcolor, tag);
        }
      }
    }
  } while (n_collapses != 0 && sweep < MAX_SWEEPS);
  owned_free(vcolor);
  return n_collapses == 0;
}

/*
 * Coarsens the boundary of the domain that holds the seed triangles (see the top of this file).
 *
 * n_points, x:    the sampled boundary points (2 coordinates each); modified.
 * tag:            the tag lists of the points (see merge_tag_list); modified.
 * n_tri, tri:     their Delaunay triangulation (3 point indices each); modified.
 * n_seeds, seeds: triangles inside the domain, at least one.
 * mesh_size:      the target size at each point; modified.
 * points, lines:  the result (see extract_boundaries), allocated here: the caller frees both with
 *                 libcfree.
 * converged:      set to 0 if the edge collapses were cut short (see optimize_mesh), else to 1.
 * Returns SEAMSH_OK, or an error code, in which case there is no result (NULL pointers).
 */
int gen_boundaries_from_points(int n_points, double *x, int *tag, int n_tri, int *tri, int n_seeds,
                               const int *seeds, double *mesh_size, double **points,
                               int *n_out_points, int **lines, int *n_lines, int *converged) {
  *points = NULL;
  *lines = NULL;
  *n_out_points = 0;
  *n_lines = 0;
  n_owned = 0;
  max_steps = 3 * (long)n_tri + 3;
  int error = setjmp(error_jump);
  if (error != SEAMSH_OK) {
    owned_free_all();
    *points = NULL;
    *lines = NULL;
    *n_out_points = 0;
    *n_lines = 0;
    return error;
  }
  if (n_seeds < 1)
    fail(SEAMSH_BAD_TRIANGULATION);
  for (int s = 0; s < n_seeds; ++s) {
    if (seeds[s] < 0 || seeds[s] >= n_tri)
      fail(SEAMSH_BAD_TRIANGULATION);
  }
  int *neigh = build_neighbours(n_tri, tri);
  orient_triangles(x, n_tri, tri, neigh);
  int *tri_color = color_triangles(x, n_tri, tri, neigh, n_seeds, seeds, mesh_size);
  *converged = optimize_mesh(x, mesh_size, n_points, n_tri, tri, tri_color, neigh, tag);
  extract_boundaries(x, mesh_size, tag, n_tri, tri, tri_color, points, n_out_points, lines,
                     n_lines);
  owned_free(tri_color);
  owned_free(neigh);
  owned_release(*points);
  owned_release(*lines);
  return SEAMSH_OK;
}

void libcfree(void *p) { free(p); }
