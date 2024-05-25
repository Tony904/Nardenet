#include "list.h"
#include "xallocs.h"



node* new_node(void* item) {
	node* n = (node*)xcalloc(1, sizeof(node));
	n->val = item;
	return n;
}

void list_append(list* lst, void* item) {
	node* n = new_node(item);
	if (lst->length == 0) {
		lst->first = n;
		lst->last = n;
		lst->length++;
		return;
	}
	lst->last->next = n;
	n->prev = lst->last;
	lst->last = n;
	lst->length++;
}

void* list_get_item(list* lst, size_t index) {
	if (!(lst->length > index)) return NULL;
	node* n = lst->first;
	for (size_t i = 0; i < index; i++) {
		n = n->next;
	}
	return n->val;
}

void free_list(list* lst) {
	node* n = lst->first;
	node* next;
	while (n) {
		next = n->next;
		xfree(n);
		n = next;
	}
	xfree(lst);
}