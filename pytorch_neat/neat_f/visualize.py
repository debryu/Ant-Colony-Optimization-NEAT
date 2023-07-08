import logging
import warnings
import copy
import graphviz


logger = logging.getLogger(__name__)


def draw_net(genome, view=False, filename=None, node_names=None, show_disabled=False, node_colors=None, fmt='png'):
    """ This is modified code originally from: https://github.com/CodeReclaimers/neat-python """
    """ Receives a genotype and draws a neural network with arbitrary topology. """
    # Attributes for network nodes.
    if graphviz is None:
        warnings.warn("This display is not available due to a missing optional dependency (graphviz)")
        return

    if node_names is None:
        node_names = {}

    assert type(node_names) is dict

    if node_colors is None:
        node_colors = {}

    assert type(node_colors) is dict

    node_attrs = {
        'shape': 'circle',
        'fontsize': '9',
        'height': '0.2',
        'width': '0.2'}

    dot = graphviz.Digraph(format=fmt, node_attr=node_attrs)

    for connect_gene in genome.connection_genes:
        if connect_gene.is_enabled or show_disabled:
            input = connect_gene.in_node_id
            output = connect_gene.out_node_id

            a = node_names.get(input, str(input))
            b = node_names.get(output, str(output))

            style = 'solid' if connect_gene.is_enabled else 'dotted'
            color = 'green' if float(connect_gene.weight) > 0 else 'red'
            width = str(0.1 + abs(float(connect_gene.weight / 5.0)))
            dot.edge(a, b, _attributes={'style': style, 'color': color, 'penwidth': width})

    dot.render(filename, view=view)

    return dot


def draw_net_c(config, genome, view=False, filename=None, node_names=None, show_disabled=True, prune_unused=False,
             node_colors=None, fmt='png'):
    
    if graphviz is None:
        warnings.warn("This display is not available due to a missing optional dependency (graphviz)")
        return

    if node_names is None:
        node_names = {}

    assert type(node_names) is dict

    if node_colors is None:
        node_colors = {}

    assert type(node_colors) is dict

    node_attrs = {
        'shape': 'circle',
        'fontsize': '9',
        'height': '0.2',
        'width': '0.2'}

    dot = graphviz.Digraph(format=fmt, node_attr=node_attrs)

    inputs = set()
    outputs = set()
    for node_gene in genome.node_genes:
        name = str(node_gene.id)
        if node_gene.type == 'input':
            inputs.add(node_gene)
            input_attrs = {'style': 'filled',
                        'shape': 'box'}
            input_attrs['fillcolor'] = node_colors.get(node_gene, 'lightgray')
            dot.node(name, _attributes=input_attrs)

        elif node_gene.type == 'output':
            outputs.add(node_gene)
            node_attrs = {'style': 'filled'}
            node_attrs['fillcolor'] = node_colors.get(node_gene, 'lightblue')

            dot.node(name, _attributes=node_attrs)

        else:
            attrs = {'style': 'filled',
                 'fillcolor': node_colors.get(node_gene, 'white')}
            dot.node(name, _attributes=attrs)


    for connect_gene in genome.connection_genes:
        if connect_gene.is_enabled or show_disabled:
            input = connect_gene.in_node_id
            output = connect_gene.out_node_id

            a = node_names.get(input, str(input))
            b = node_names.get(output, str(output))

            style = 'solid' if connect_gene.is_enabled else 'dotted'
            color = 'green' if float(connect_gene.weight) > 0 else 'red'
            width = str(0.1 + abs(float(connect_gene.weight / 5.0)))
            dot.edge(a, b, _attributes={'style': style, 'color': color, 'penwidth': width})

    dot.render(filename, view=view)

    return dot

    """ Receives a genome and draws a neural network with arbitrary topology. """
    # Attributes for network nodes.
    if graphviz is None:
        warnings.warn("This display is not available due to a missing optional dependency (graphviz)")
        return

    if node_names is None:
        node_names = {}

    assert type(node_names) is dict

    if node_colors is None:
        node_colors = {}

    assert type(node_colors) is dict

    node_attrs = {
        'shape': 'circle',
        'fontsize': '9',
        'height': '0.2',
        'width': '0.2'}

    dot = graphviz.Digraph(format=fmt, node_attr=node_attrs)

    inputs = set()
    for k in config.genome_config.input_keys:
        inputs.add(k)
        name = node_names.get(k, str(k))
        input_attrs = {'style': 'filled',
                       'shape': 'box'}
        input_attrs['fillcolor'] = node_colors.get(k, 'lightgray')
        dot.node(name, _attributes=input_attrs)

    outputs = set()
    for k in config.genome_config.output_keys:
        outputs.add(k)
        name = node_names.get(k, str(k))
        node_attrs = {'style': 'filled'}
        node_attrs['fillcolor'] = node_colors.get(k, 'lightblue')

        dot.node(name, _attributes=node_attrs)

    if prune_unused:
        connections = set()
        for cg in genome.connections.values():
            if cg.enabled or show_disabled:
                connections.add((cg.in_node_id, cg.out_node_id))

        used_nodes = copy.copy(outputs)
        pending = copy.copy(outputs)
        while pending:
            new_pending = set()
            for a, b in connections:
                if b in pending and a not in used_nodes:
                    new_pending.add(a)
                    used_nodes.add(a)
            pending = new_pending
    else:
        used_nodes = set(genome.nodes.keys())

    for n in used_nodes:
        if n in inputs or n in outputs:
            continue

        attrs = {'style': 'filled',
                 'fillcolor': node_colors.get(n, 'white')}
        dot.node(str(n), _attributes=attrs)

    for cg in genome.connections.values():
        if cg.enabled or show_disabled:
            #if cg.input not in used_nodes or cg.output not in used_nodes:
            #    continue
            input, output = cg.key
            a = node_names.get(input, str(input))
            b = node_names.get(output, str(output))
            style = 'solid' if cg.enabled else 'dotted'
            color = 'green' if cg.weight > 0 else 'red'
            width = str(0.1 + abs(cg.weight / 5.0))
            dot.edge(a, b, _attributes={'style': style, 'color': color, 'penwidth': width})

    dot.render(filename, view=view)

    return dot
