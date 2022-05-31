
import numina.core

from ..loader import drp_load


def test_recipes_are_defined():

    this_drp = drp_load()

    assert 'default' in this_drp.pipelines

    for pipeval in this_drp.pipelines.values():
        for key, val in pipeval.recipes.items():
            recipe = pipeval.get_recipe_object(key)
            assert isinstance(recipe, numina.core.BaseRecipe)