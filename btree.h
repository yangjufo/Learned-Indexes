/* btree.h - A binary tree implementation */

/*
Public functions
----------------

BTREE btree_Create(size_t size, int (*cmp)(const void *, const void *))

Create a btree which holds elements of size <size>.  The (non-strict)
ordering or the elements is determined by the function <cmp>.  <cmp> should
be a function which takes a two pointers to elements and returns a value
less than, equal to, or greater than zero if the first value is less than,
equal to, or greater than the second, respectively.


int btree_Search(BTREE tree, void *key, void *ret)

Search the <tree> for an element "equal" to <key> and return it in the space
pointed to by <ret>.  If <ret> is NULL, then don't bother returning any
values.  This routine returns a zero value on success and a non-zero value
if it fails to find the <key> value.


int btree_Minimum(BTREE tree, void *ret)
int btree_Maximum(BTREE tree, void *ret)

Return the minimum/maximum value in <tree> in the space pointed to by <ret>.
If <ret> is NULL, then don't bother returning any values.  This routine
returns a zero value on success and a non-zero value if it fails, (i.e., the
tree is empty).


int btree_Empty(BTREE tree)

Returns a non-zero value if the <tree> is empty, and a zero value if it
contains elements.


int btree_Successor(BTREE tree, void *key, void *ret)
int btree_Predecessor(BTREE tree, void *key, void *ret)

Returns the value immediately after/before <key> in the <tree>.  The value
is returned in the space pointed to by <ret>.  If <ret> is NULL, then don't
bother to return anything.  This routine returns a zero value on success and
a non-zero value if it fails.


int btree_Insert(BTREE tree, void *elem)

Insert <elem> into the <tree>.  Returns zero on success and non-zero if it
fails.


int btree_Delete(BTREE tree, void *elem)

Remove <elem> from the <tree>.  Returns zero on success and non-zero if it
fails, i.e., <elem> is not in the <tree> to begin with.


void btree_Destroy(BTREE tree)

Close <tree> and free all of the memory that it used.
*/


#ifndef BTREE_H_INC
#define BTREE_H_INC

/* ---------- Incomplete data types ---------- */
typedef struct _btree_struct *BTREE;


/* ---------- Public function declarations ---------- */
extern BTREE btree_Create(size_t size, int(*cmp)(const void *, const void *));
extern int btree_Search(BTREE tree, void *key, void *ret);
extern int btree_Minimum(BTREE tree, void *ret);
extern int btree_Maximum(BTREE tree, void *ret);
extern int btree_Empty(BTREE tree);
extern int btree_Successor(BTREE tree, void *key, void *ret);
extern int btree_Predecessor(BTREE tree, void *key, void *ret);
extern int btree_Insert(BTREE tree, void *data);
extern int btree_Delete(BTREE tree, void *data);
extern void btree_Destroy(BTREE);
extern void btree_print(BTREE, void *);	/* Do NOT use. */
#endif /* BTREE_H_INC */