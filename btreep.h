#pragma once
/* btreepriv.h - Private declarations for btrees */

#ifndef BTREEPRIV_H_INC
#define BTREEPRIV_H_INC

/* ---------- Private data type declarations ---------- */
struct _btree_struct {
	struct node *root;
	size_t node_size;		/* Size of the node is unknown, e.g., R-B trees. */
	size_t elem_size;
	int(*cmp)(const void *, const void *);
};

typedef struct node {
	struct node *parent;
	struct node *left;
	struct node *right;
} *btnode;

/* ---------- Private macros ---------- */
#define root(t) (t->root)
#define node_size(t) (t->node_size)
#define elem_size(t) (t->elem_size)

#define parent(n) (n->parent)
#define left(n) (n->left)
#define right(n) (n->right)

#define data(t,n) (((char *)n) + node_size(t))
#define data_copy(t, d, s)      memcpy(d, s, elem_size(t))
#define data_compare(t, f, s)   ((t)->cmp)(f, s)

#endif /* BTREEPRIV_H_INC */